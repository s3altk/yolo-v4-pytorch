import torch
from torch.autograd import Variable

import os, sys
import cv2
import time
import math
import itertools
import struct
import imghdr
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def do_detect(model, img, conf_thresh, n_classes, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()
    
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
        
    elif type(img) == np.ndarray and len(img.shape) == 3:
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)

    else:
        print('Неизвестный тип изображения')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    t2 = time.time()
    
    list_features = model(img)
    list_features_numpy = []

    for feature in list_features:
        list_features_numpy.append(feature.data.cpu().numpy())

    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    anchor_step = len(anchors) // num_anchors
    boxes = []

    for i in range(3):
        masked_anchors = []        
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]

        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
        boxes.append(get_region_boxes_out_model(list_features_numpy[i], conf_thresh, n_classes, masked_anchors, len(anchor_masks[i])))

    if img.shape[0] > 1:
        bboxs_for_imgs = [boxes[0][index] + boxes[1][index] + boxes[2][index] for index in range(img.shape[0])]
        t3 = time.time()
        boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]

    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        t3 = time.time()
        boxes = nms(boxes, nms_thresh)

    t4 = time.time()

    print('----------------------------------------')
    print('     Классификация : %f' % (t3 - t2))
    print('    Детектирование : %f' % (t4 - t3))
    print('             Всего : %f' % (t4 - t0))
    print('----------------------------------------')

    return boxes


def get_region_boxes_out_model(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors

    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)

    batch = output.shape[0]

    assert (output.shape[1] == (5 + num_classes) * num_anchors)

    h = output.shape[2]
    w = output.shape[3]

    all_boxes = []
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose((1, 0, 2)).reshape(5 + num_classes, batch * num_anchors * h * w)

    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(batch * num_anchors, axis=0).reshape(batch * num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(batch * num_anchors, axis=0).reshape(batch * num_anchors * h * w)

    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = np.array(anchors).reshape((num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(anchors).reshape((num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2).repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2).repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)

    ws = np.exp(output[2]) * anchor_w
    hs = np.exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])
    cls_confs = softmax(output[5:5 + num_classes].transpose(1, 0))
    cls_max_confs = np.max(cls_confs, 1)
    cls_max_ids = np.argmax(cls_confs, 1)

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]

                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]

                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]

                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]

                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)

                        boxes.append(box)

        all_boxes.append(boxes)

    return all_boxes

def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)

def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]

        if box_i[4] > 0:
            out_boxes.append(box_i)
            
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]

                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0

    return out_boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])

        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)

        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

    uw = Mx - mx
    uh = My - my

    cw = w1 + w2 - uw
    ch = h1 + h2 - uh

    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2

    carea = cw * ch
    uarea = area1 + area2 - carea

    return carea / uarea


def load_class_names(namesfile):
    class_names = []

    with open(namesfile, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.rstrip()
        class_names.append(line)

    return class_names


def plot_boxes_cv2(img_file, boxes, save_img_file=None, class_names=None, color=None):
    img = cv2.imread(img_file)

    height = img.shape[0]
    width = img.shape[1]
    
    for i in range(len(boxes)):
        box = boxes[i]

        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (0, 0, 255) # синий, зеленый, красный
        
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            cls_name = class_names[cls_id]

            img = cv2.putText(img, f'{cls_name} {cls_conf:>4f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1, cv2.LINE_AA)
            print(f'Найден объект класса {cls_name} с вероятностью {cls_conf:>4f}')
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 5)

    if save_img_file:
        cv2.imwrite(save_img_file, img)
        print(f'Размеченное изображение сохранено в {save_img_file}')

    return img

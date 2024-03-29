import json
import tempfile

import numpy as np
import copy
import time
import torch
import torch._six

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from collections import defaultdict

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, bbox_fmt='coco'):
        assert isinstance(iou_types, (list, tuple))

        coco_gt = copy.deepcopy(coco_gt)

        self.coco_gt = coco_gt
        self.bbox_fmt = bbox_fmt.lower()

        assert self.bbox_fmt in ['voc', 'coco', 'yolo']

        self.iou_types = iou_types
        self.coco_eval = {}

        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))

        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)

            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print('IoU: {}'.format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == 'bbox':
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError('Неизвестный тип IoU: {}'.format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []

        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            
            if self.bbox_fmt == 'coco':
                boxes = prediction['boxes'].tolist()
            else:
                boxes = prediction['boxes']
                boxes = convert_to_xywh(boxes, fmt=self.bbox_fmt).tolist()

            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            coco_results.extend(
                [
                    {
                        'image_id': original_id,
                        'category_id': labels[k],
                        'bbox': box,
                        'score': scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes, fmt='voc'):
    if fmt.lower() == 'voc':
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    elif fmt.lower() == 'yolo':
        xcen, ycen, w, h = boxes.unbind(1)
        return torch.stack((xcen-w/2, ycen-h/2, w, h), dim=1)

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)

    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def loadRes(self, resFile):
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))

    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
        
    else:
        anns = resFile

    assert type(anns) == list, 'результат не найден'

    annsImgIds = [ann['image_id'] for ann in anns]

    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'результат не удается привести к датасету COCO'

    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])

        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]

        for id, ann in enumerate(anns):
            ann['id'] = id + 1

    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

        for id, ann in enumerate(anns):
            ann['bbox'] = ann['bbox'][0]
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]

            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0

    res.dataset['annotations'] = anns

    createIndex(res)

    return res

def createIndex(self):
    anns, cats, imgs = {}, {}, {}

    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


def evaluate(self):
    p = self.params
    p.imgIds = list(np.unique(p.imgIds))

    if p.useCats:
        p.catIds = list(np.unique(p.catIds))

    p.maxDets = sorted(p.maxDets)

    self.params = p
    self._prepare()

    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'bbox':
        computeIoU = self.computeIoU

    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]

    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

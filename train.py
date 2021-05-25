import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import logging
import os, sys
import argparse

from tqdm import tqdm
from cfg import Cfg
from models import Yolov4
from dataset import Yolo_dataset
from easydict import EasyDict as edict


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device=device)

    logging = init_logger(log_dir='log')
    logging.info(f'Используется устройство: {device}')

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device,
              save_dir=cfg.save_dir)

    except KeyboardInterrupt:
        interrupt_dir = 'yolov4.train.interrupt.pth'

        torch.save(model.state_dict(), interrupt_dir)
        logging.info(f'Текущая модель сохранена в {interrupt_dir}')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def get_args(**kwargs):
    parser = argparse.ArgumentParser(description='Обучение модели на изображениях', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Размер батча', dest='batch')
    parser.add_argument('-s', '--subdivisions', metavar='S', type=int, nargs='?', default=1, help='Размер мини-батча', dest='subdivisions')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Скорость обучения', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,  help='Загрузить модель из .pth файла')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1', help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None, help='Директория датасета', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='Предобученная модель')
    parser.add_argument('-save_dir', type=str, default=None, help='Директория для сохранения модели')
    parser.add_argument('-classes', type=int, default=80, help='Количество классов')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="Наименования классов")
    parser.add_argument('-epochs', dest='TRAIN_EPOCHS', type=int, default=10, help="Количество эпох")
    
    args = vars(parser.parse_args())
    cfg = kwargs

    for k in args.keys():
        cfg[k] = args.get(k)

    return edict(cfg)

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    import datetime

    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    if log_dir is None:
        log_dir = '~/temp/log/'

    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_file)

    print(f'Файл логов находится в {log_file}')
    
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)

        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)

        logging.getLogger('').addHandler(console)

    return logging

def train(model, device, config, save_dir, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config)
    test_dataset = Yolo_dataset(config.val_label, config)

    n_train = len(train_dataset)
    n_test = len(test_dataset)

    logging.info(f'''Запуск обучения:
            Кол-во эпох:                    {epochs}
            Размер партии:                  {config.batch}
            Размер мини-партии:             {config.subdivisions}
            Скорость обучения:              {config.learning_rate}
            Размер обуч. выборки:           {n_train}
            Размер тест. выборки:           {n_test}
            Включить сохранение модели:     {save_cp}
            Устройство:                     {device.type}
            Кол-во изображений:             {config.width}
            Оптимизатор:                    {config.TRAIN_OPTIMIZER}
            Кол-во классов:                 {config.classes}
            Названия классов (файл):        {config.train_label}
    ''')

    def collate(batch):
        images = []
        bboxes = []

        for img, box in batch:
            images.append([img])
            bboxes.append([box])

        images = np.concatenate(images, axis=0)
        images = images.transpose(0, 3, 1, 2)
        images = torch.from_numpy(images).div(255.0)

        bboxes = np.concatenate(bboxes, axis=0)
        bboxes = torch.from_numpy(bboxes)

        return images, bboxes

    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01

        return factor

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate / config.batch, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    loss_fn = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    model.train()

    for epoch in range(epochs):
        logging.info(f'Эпоха {epoch + 1}\n-------------------------------')
        
        with tqdm(total=n_train, desc=f'[{(epoch + 1):>3d}/{epochs:>3d}]', unit='img', ncols=50) as pbar:
            for batch, (X, y) in enumerate(train_loader):
                images, bboxes = X.to(device=device, dtype=torch.float32), y.to(device=device)
   
                bboxes_pred = model(images)

                loss = loss_fn(bboxes_pred, bboxes)
                loss.backward()

                if batch  % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if batch % (log_step * config.subdivisions) == 0:
                    loss, current = loss.item(), batch * len(images)
                    lr = scheduler.get_lr()[0] * config.batch
                    logging.info(f'[{current:>5d}/{n_train:>5d}] --> функция потерь={loss:>7f}, скорость обучения={lr:>7f}')

                pbar.update(images.shape[0])

        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in enumerate(test_loader):
                images, bboxes = X.to(device=device, dtype=torch.float32), y.to(device=device)

                bboxes_pred = model(images)
                
                test_loss += loss_fn(bboxes_pred, bboxes).item()
                correct += (bboxes_pred.argmax(1) == bboxes).type(torch.float).sum().item()

        test_loss /= n_test
        correct /= n_test

        logging.info(f'Результаты --> сред.знач. точности={(100 * correct):>0.1f}, сред.знач. функции потерь={test_loss:>7f}\n')

        if save_cp:
            try:
                os.mkdir(save_dir)
                logging.info(f'Создана директория {save_dir} для сохранения')
            except OSError:
                pass

            torch.save(model.state_dict(), os.path.join(save_dir, f'yolov4.train.epoch{epoch + 1}.pth'))
            logging.info(f'Модель эпохи {epoch + 1} сохранена!')

    logging.info(f'Обучение завершено!')


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        image_size = 608

        self.device = device
        self.strides = [8, 16, 32]
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id])
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2], weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    return area_i / (area_a[:, None] + area_b - area_i)

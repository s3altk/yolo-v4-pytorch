import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

def convert_to_coco_api(ds, bbox_fmt='voc'):
    coco_ds = COCO()

    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}

    categories = set()

    for img_idx in range(len(ds)):
        img, targets = ds[img_idx]

        image_id = targets['image_id'].item()

        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]

        dataset['images'].append(img_dict)
        bboxes = targets['boxes']

        if bbox_fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
            bboxes[:, 2:] -= bboxes[:, :2]

        elif bbox_fmt.lower() == 'yolo':  # xcen, ycen, w, h
            bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:]/2

        elif bbox_fmt.lower() == 'coco':
            pass
        else:
            raise ValueError(f'Выбранный формат {bbox_fmt} не поддерживается!')

        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()

        if 'masks' in targets:
            masks = targets['masks']
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)

        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()

        num_objs = len(bboxes)

        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            
            categories.add(labels[i])

            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id

            if 'masks' in targets:
                ann['segmentation'] = coco_mask.encode(masks[i].numpy())

            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])

            dataset['annotations'].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]

    coco_ds.dataset = dataset
    coco_ds.createIndex()

    return coco_ds

import torch.nn as nn
import torch.nn.functional as F

def get_targets(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1, validation=False):
    bxy_list = []
    bwh_list = []
    
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    bxy = torch.cat(bxy_list, dim=1)
    bwh = torch.cat(bwh_list, dim=1)

    det_confs = torch.cat(det_confs_list, dim=1)
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    cls_confs = torch.cat(cls_confs_list, dim=1)
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)
    
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)

    anchor_w = []
    anchor_h = []
    
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []
    
    for i in range(num_anchors):
        ii = i * 2
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32)
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32)
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    bx = torch.cat(bx_list, dim=1)
    by = torch.cat(by_list, dim=1)
    bw = torch.cat(bw_list, dim=1)
    bh = torch.cat(bh_list, dim=1)

    bx_bw = torch.cat((bx, bw), dim=1)
    by_bh = torch.cat((by, bh), dim=1)

    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)

    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    confs = cls_confs * det_confs

    return boxes, confs

class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output):
        masked_anchors = []
        
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
            
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return get_targets(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)

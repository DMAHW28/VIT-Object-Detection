import torch
import torch.nn as nn

def bbox_iou(box1, box2, eps=1e-7):
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter_area + eps

    return inter_area / union

def bbox_giou(box1, box2, eps=1e-7):
    iou = bbox_iou(box1, box2, eps)

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])

    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_area = iou * (area1 + area2 - iou)
    union = area1 + area2 - inter_area

    giou = iou - ((c_area - union) / (c_area + eps))
    return giou

def bbox_diou(box1, box2, eps=1e-7):
    iou = bbox_iou(box1, box2, eps)

    b1_xc = (box1[:, 0] + box1[:, 2]) / 2
    b1_yc = (box1[:, 1] + box1[:, 3]) / 2
    b2_xc = (box2[:, 0] + box2[:, 2]) / 2
    b2_yc = (box2[:, 1] + box2[:, 3]) / 2

    center_dist = (b1_xc - b2_xc) ** 2 + (b1_yc - b2_yc) ** 2

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    diou = iou - center_dist / diag
    return diou

# =====================
#   Loss Wrappers
# =====================
class IoULoss(nn.Module):
    def forward(self, pred_boxes, target_boxes):
        return 1 - bbox_iou(pred_boxes, target_boxes).mean()

class GIoULoss(nn.Module):
    def forward(self, pred_boxes, target_boxes):
        return 1 - bbox_giou(pred_boxes, target_boxes).mean()

class DIoULoss(nn.Module):
    def forward(self, pred_boxes, target_boxes):
        return 1 - bbox_diou(pred_boxes, target_boxes).mean()

if __name__ == "__main__":
    box1 = torch.tensor([[0., 0., 2., 2.]])
    box2 = torch.tensor([[1., 1., 3., 3.]])

    loss_fn = DIoULoss()
    loss = loss_fn(box1, box2)
    print("DIoU Loss:", loss.item())

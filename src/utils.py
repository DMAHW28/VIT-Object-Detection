import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.loader import LABEL_KEYS

INV_LABEL_KEYS = {v: k for k, v in LABEL_KEYS.items()}

def denormalize(image):
    img = (image * 0.5 + 0.5).clip(0, 1)
    return img

def draw_boxes_on_tensor(img_tensor, pred_box, pred_cls, gt_box=None, gt_cls=None, with_denorm=False):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    if with_denorm:
        img = denormalize(img)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')

    xmin, ymin, xmax, ymax = pred_box
    rect = patches.Rectangle((xmin*img.shape[1], ymin*img.shape[0]), (xmax-xmin)*img.shape[1], (ymax-ymin)*img.shape[0], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin*img.shape[1], ymin*img.shape[0]-5, f"Pred: {INV_LABEL_KEYS[pred_cls]}", color='r', fontsize=8, weight="bold")

    if gt_box is not None:
        xmin, ymin, xmax, ymax = gt_box
        rect = patches.Rectangle((xmin*img.shape[1], ymin*img.shape[0]), (xmax-xmin)*img.shape[1], (ymax-ymin)*img.shape[0], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin*img.shape[1], ymax*img.shape[0]+10, f"GT: {INV_LABEL_KEYS[int(gt_cls)]}", color='g', fontsize=8, weight="bold")

    fig.canvas.draw()
    img_with_boxes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()  # <-- fix warning
    img_with_boxes = img_with_boxes.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img_with_boxes = torch.from_numpy(img_with_boxes).permute(2, 0, 1).float() / 255.
    return img_with_boxes

def make_grid_with_boxes(imgs, pred_boxes, pred_classes, gt_boxes=None, gt_classes=None, n_row=4):
    img_list = []
    for i in range(len(imgs)):
        gt_b = gt_boxes[i] if gt_boxes is not None else None
        gt_c = gt_classes[i] if gt_classes is not None else None
        img_with_boxes = draw_boxes_on_tensor(imgs[i], pred_boxes[i], pred_classes[i], gt_b, gt_c)
        img_list.append(img_with_boxes)
    grid = torchvision.utils.make_grid(img_list, nrow=n_row)
    return grid
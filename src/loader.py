import torch
import scipy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

I_SIZE = 224
LABEL_KEYS = {
    "airplanes": 0,
    "helicopter": 1,
    "Motorbikes": 2,
}

train_transform = A.Compose([
    A.Resize(I_SIZE, I_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(0.05, 0.05),
        rotate=(-10, 10),
        shear=0,
        border_mode=0,
        p=0.5
    ),
    A.ColorJitter(p=0.5),
    A.MotionBlur(p=0.2),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

test_transform = A.Compose([
    A.Resize(I_SIZE, I_SIZE),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def read_data(data_file):
    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(line.strip().split(','))
    return data[1:]

class AirPlaneDetectionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, box_file, name = self.data[idx]
        img = np.array(Image.open(img_file).convert("RGB"))
        annot = scipy.io.loadmat(box_file)["box_coord"][0]
        ymin, ymax, xmin, xmax = annot

        transformed = self.transform(
            image=img,
            bboxes=[[xmin, ymin, xmax, ymax]],
            class_labels=[LABEL_KEYS[name]]
        )
        img = transformed["image"]
        box = transformed["bboxes"][0]
        cls = transformed["class_labels"][0]

        h, w = img.shape[1], img.shape[2]  # C,H,W
        xmin, ymin, xmax, ymax = box
        xmin_norm, ymin_norm = xmin / w, ymin / h
        xmax_norm, ymax_norm = xmax / w, ymax / h
        box = torch.tensor([xmin_norm, ymin_norm, xmax_norm, ymax_norm], dtype=torch.float32)
        cls = torch.tensor([cls]).long()
        return img, cls, box

class AirPlaneData:
    def __init__(self, train_txt="database/train.txt", test_txt="database/test.txt"):
        self.train_data = read_data(train_txt)
        self.test_data = read_data(test_txt)

    def __call__(self, batch_size=16, num_workers=0, train_transf=train_transform, test_transf=test_transform):
        train_dataset = AirPlaneDetectionDataset(self.train_data, transform=train_transf)
        test_dataset = AirPlaneDetectionDataset(self.test_data, transform=test_transf)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader

# /Users/liamdro/PycharmProjects/AirPlaneDetection/database/101_ObjectCategories/airplanes/image_0118.jpg airplanes

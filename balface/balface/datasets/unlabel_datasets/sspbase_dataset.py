import os
import os.path as osp
import random
import numpy as np


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

from balface.utils import GaussianBlur, TwoCropsTransform




class SSPBaseDataset(Dataset):

    def __init__(self, root, face_bbox_txt, size=224, *args, **kwargs):
        super(SSPBaseDataset, self).__init__()
        self.root = root
        self.size = size

        bbox_labels = {}
        with open(face_bbox_txt, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                image_name = line[0]
                bbox = np.array(list(map(int, line[1:])), dtype=np.int32) # (x0, y0, x1, y1)
                bbox_labels[image_name] = bbox
        self.bbox_labels = bbox_labels
        self.image_list = list(bbox_labels.keys())
        self.image_list.sort()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        augmentation = [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        self.pad_ratio = 0.25

        self.transform = TwoCropsTransform(transforms.Compose(augmentation))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        path = osp.join(self.root, self.image_list[index])

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        # crop face with padding from image
        w, h = sample.size
        x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
        face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
        face_pad_w, face_pad_h = int(face_w * self.pad_ratio / 2), int(face_h * self.pad_ratio / 2)
        x0, y0, x1, y1 = x0 - face_pad_w, y0 - face_pad_h, x1 + face_pad_w, y1 + face_pad_h
        x0 = max(0, x0)
        x1 = min(w, x1)
        y0 = max(0, y0)
        y1 = min(h, y1)
        sample = sample.crop((x0, y0, x1, y1))

        sample_a, sample_b = self.transform(sample)

        return sample_a, sample_b
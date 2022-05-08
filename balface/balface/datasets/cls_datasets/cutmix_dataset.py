import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur



class CutmixFairFace4RacesDataset(Dataset):
    ATTRS_NUM = {
        'age': 9,
        'gender': 2,
        'race': 4
    }

    ATTRS_LIST = {
        'race': {
            'White': 0,
            'Black': 1,
            'East Asian': 2,
            'Indian': 3
        },
        'gender': {
            'Male': 0,
            'Female': 1
        },
        'age': {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            '70+': 8
        }
    }

    ATTRS_REV_INDEX = {
        'race': {
            0: 'White',
            1: 'Black',
            2: 'East Asian',
            3: 'Indian'
        },
        'gender': {
            0: 'Male',
            1: 'Female'
        },
        'age': {
            0: '0-2',
            1: '3-9',
            2: '10-19',
            3: '20-29',
            4: '30-39',
            5: '40-49',
            6: '50-59',
            7: '60-69',
            8: '70+'
        }
    }


    def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, aug_num=0, *args, **kwargs):
        super(CutmixFairFace4RacesDataset, self).__init__()

        self.root = root
        self.size = size
        self.ssp = ssp

        bbox_labels = {}
        with open(face_bbox_txt, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                image_name = line[0]
                bbox = np.array(list(map(int, line[1:])), dtype=np.int32) # (x0, y0, x1, y1)
                image_name = image_name.replace('images', '025_images')
                bbox_labels[image_name] = bbox

        attr_labels = {}
        self.image_list = []
        self.bbox_labels = {}

        self.class_wise_list = [[], [], [], []]

        with open(face_label_txt, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i==0:
                    attrs = line.strip().split(',')[1:]
                    for attr in attrs:
                        assert attr in self.ATTRS_NUM
                    self.attrs = attrs
                    continue
                line = line.strip().split(',')
                image_name = line[0]
                image_name.replace('images', '025_images')
                labels = []

                if line[1] == 'Middle Eastern': line[1] = 'White'
                if line[1] == 'Southeast Asian': line[1] = 'East Asian'
                if line[1] in ['Latino_Hispanic', 'Middle Eastern'] : continue
                for i in range(len(self.attrs)):
                    attr = self.attrs[i]
                    name = line[i+1]
                    labels.append(self.ATTRS_LIST[attr][name])

                labels = np.array(labels).astype(np.int32)
                attr_labels[image_name] = labels
                assert bbox_labels.get(image_name) is not None

                self.bbox_labels[image_name] = bbox_labels[image_name]
                self.image_list.append(image_name)
                self.class_wise_list[self.ATTRS_LIST['race'][line[1]]].append(image_name)

        self.image_list.sort()
        self.attr_labels = attr_labels
        self.attr_cls_num = [self.ATTRS_NUM[attr] for attr in self.attrs]

        self.attr_rev_mapping = []
        for attr in self.attrs:
            self.attr_rev_mapping.append(self.ATTRS_REV_INDEX[attr])

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if mode == 'train':
            augmentation = [
                transforms.RandomRotation(degrees=30),
                # transforms.RandomCrop(size),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.)), # ratio=(1.0, 1.0)
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
            self.pad_ratio = 2.0
        else:
            augmentation = [
                transforms.Resize((size, size)),
                # transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
            self.pad_ratio = 1.0

        self.transform = transforms.Compose(augmentation)
        self.mode = mode
        self.aug_num = aug_num

        cls_nums = np.array([len(self.class_wise_list[i]) for i in range(4)]).astype(np.float32)
        sample_ratio = 1./cls_nums
        self.sample_ratio = sample_ratio / (sample_ratio.sum() + 1e-6)
        self.sample_nums = (aug_num * self.sample_ratio).astype(np.int32)

        self.max_coeff = 0.1

    def __len__(self):
        return len(self.image_list) + self.sample_nums.sum()

    def sample_cutmix(self):
        cls1, cls2 = np.random.choice(4, 2, replace=True, p=self.sample_ratio)
        img_name1_index, img_name2_index = np.random.randint(0, len(self.class_wise_list[cls1]), 1)[0], np.random.randint(0, len(self.class_wise_list[cls2]), 1)[0]
        img_name1 = self.class_wise_list[cls1][img_name1_index]
        img_name2 = self.class_wise_list[cls2][img_name2_index]

        coeff = np.random.uniform(0., self.max_coeff)
        return cls1, cls2, img_name1, img_name2, coeff

    def mix_images(self, pil_img1, pil_img2, coeff):
        rw = int(self.size * np.sqrt(1-coeff))
        rh = rw

        rx = np.random.randint(self.size+1-rw)
        ry = np.random.randint(self.size+1-rh)

        cropped = pil_img2.crop((rx, ry, rx+rw, ry+rh))
        pil_img1.paste(cropped, (rx, ry, rx+rw, ry+rh))

        return pil_img1

    def __getitem__(self, index):
        assert self.mode == 'train'

        if index < len(self.image_list):

            path = osp.join(self.root, self.image_list[index])
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')

            if self.mode == 'val':
                # crop face with padding from image
                w, h = sample.size
                x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
                center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
                face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
                lateral = max(face_w, face_h)
                lateral = int(lateral * (1 + self.pad_ratio))

                x0, x1 = center_w - lateral // 2, center_w + lateral // 2
                y0, y1 = center_h - lateral // 2, center_h + lateral // 2

                sample = sample.crop((x0, y0, x1, y1))

            face = self.transform(sample)

            labels = self.attr_labels[self.image_list[index]]

            labels = torch.from_numpy(labels).long()

            if self.ssp:
                face2 = self.transform(sample)
                return face, face2, labels

            labels = F.one_hot(labels[0], num_classes=4).float()

            return face, labels

        else:
            cls1, cls2, img_name1, img_name2, coeff = self.sample_cutmix()

            path = osp.join(self.root, img_name1)
            with open(path, 'rb') as f:
                sample1 = Image.open(f).convert('RGB')

            path = osp.join(self.root, img_name2)
            with open(path, 'rb') as f:
                sample2 = Image.open(f).convert('RGB')

            sample = self.mix_images(sample1, sample2, coeff)
            face = self.transform(sample)

            labels1 = F.one_hot(torch.tensor(cls1).long(), num_classes=4).float()
            labels2 = F.one_hot(torch.tensor(cls2).long(), num_classes=4).float()

            label = labels1 * coeff + labels2 * (1-coeff)

            return face, label






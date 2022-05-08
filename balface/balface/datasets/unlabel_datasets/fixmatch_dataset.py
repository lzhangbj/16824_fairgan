import os
import os.path as osp
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

from balface.utils import GaussianBlur, TwoCropsTransform


class FixMatchDataset(Dataset):

	def __init__(self, root, face_bbox_txt, size=224, mode='train', ssp=False, *args, **kwargs):
		super(FixMatchDataset, self).__init__()
		assert mode in ['train','val'], mode
		self.root = root
		self.size = size
		self.ssp = ssp

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

		self.original_image_list = self.image_list.copy()

		self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

		self.mode = mode
		if mode == 'train':
			self.weak_transform = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				# transforms.RandomResizedCrop(size, scale=(0.8, 0.8), ratio=(1., 1.)),
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			])

			self.strong_transform = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(degrees=30),
				transforms.RandomResizedCrop(size, scale=(0.5, 1.)),
				# transforms.RandAugment(2, 10),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			])
			self.pad_ratio = 0.25
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]

			self.pad_ratio = 0

			self.transform = transforms.Compose(augmentation)

		self.val_transform = transforms.Compose([
			transforms.Resize((size, size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.mean, std=self.std)
		])

		self.attr_labels = None

		self.gen_pseudolabel = False


	def update_image_list(self, image_list):
		self.image_list = image_list

	# def refresh_pseudolabels(self, label_list, attr_list):
	#     attr_labels = {}
	#     for i, line in enumerate(label_list):
	#         if i==0:
	#             attrs = line.strip().split(',')[1:]
	#             self.attrs = attrs
	#             continue
	#         line = line.strip().split(',')
	#         image_name = line[0]
	#         labels = []
	#         for i in range(len(self.attrs)):
	#             attr = self.attrs[i]
	#             name = line[i+1]
	#             labels.append(attr_list[attr][name])
	#         labels = np.array(labels).astype(np.int32)
	#         attr_labels[image_name] = labels
	#     curr_image_names = list(attr_labels.keys())
	#     curr_image_names.sort()
	#     assert self.image_list == curr_image_names, [len(self.image_list), len(curr_image_names)]
	#     self.attr_labels = attr_labels

	def __len__(self):
		if self.gen_pseudolabel:
			return len(self.original_image_list)
		return len(self.image_list)

	def __getitem__(self, index):
		if self.gen_pseudolabel:
			image_name = self.original_image_list[index]
			path = osp.join(self.root, self.original_image_list[index])
		else:
			image_name = self.image_list[index]
			path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		w, h = sample.size
		x0, y0, x1, y1 = self.bbox_labels[image_name]
		face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		lateral = max(face_w, face_h)
		face_pad = int(lateral * self.pad_ratio / 2)
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio / 2), int(face_h * self.pad_ratio / 2)
		x0, y0, x1, y1 = x0 - face_pad, y0 - face_pad, x1 + face_pad, y1 + face_pad
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)
		sample = sample.crop((x0, y0, x1, y1))

		# sample.save(f'./pseudo_images/{self.image_list[index]}')
		if self.mode == 'val':
			face = self.transform(sample)
			return face, 0

		weak_sample = self.weak_transform(sample)

		if self.gen_pseudolabel:
			return weak_sample, index

		strong_sample = self.strong_transform(sample)

		# if self.attr_labels is not None:
		#     labels = self.attr_labels[self.image_list[index]]
		#     labels = torch.from_numpy(labels).long()
		#     return sample, index, labels

		if self.ssp:
			strong_sample2 = self.strong_transform(sample)
			return weak_sample, strong_sample, strong_sample2

		return weak_sample, strong_sample


class FixMatch025Dataset(Dataset):

	def __init__(self, root, face_label_txt, size=224, mode='train', ssp=False, *args, **kwargs):
		super(FixMatch025Dataset, self).__init__()
		assert mode in ['train','val'], mode
		self.root = root
		self.size = size
		self.ssp = ssp

		self.image_list = []
		with open(face_label_txt, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i == 0:
					continue
				line = line.strip().split(',')
				# if line[1] == 'White': continue
				image_name = line[0]
				self.image_list.append(image_name)
		self.image_list.sort()
		self.image_list = self.image_list
		self.original_image_list = self.image_list.copy()

		self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

		self.mode = mode
		if mode == 'train':
			self.weak_transform = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				# transforms.RandomResizedCrop(size, scale=(0.8, 0.8), ratio=(1., 1.)),
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			])

			self.strong_transform = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(degrees=30),
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),
				transforms.RandAugment(2, 10),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			])
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]

			self.transform = transforms.Compose(augmentation)

		self.val_transform = transforms.Compose([
			transforms.Resize((size, size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.mean, std=self.std)
		])

		self.attr_labels = None

		self.gen_pseudolabel = False


	def update_image_list(self, image_list):
		self.image_list = image_list

	# def refresh_pseudolabels(self, label_list, attr_list):
	#     attr_labels = {}
	#     for i, line in enumerate(label_list):
	#         if i==0:
	#             attrs = line.strip().split(',')[1:]
	#             self.attrs = attrs
	#             continue
	#         line = line.strip().split(',')
	#         image_name = line[0]
	#         labels = []
	#         for i in range(len(self.attrs)):
	#             attr = self.attrs[i]
	#             name = line[i+1]
	#             labels.append(attr_list[attr][name])
	#         labels = np.array(labels).astype(np.int32)
	#         attr_labels[image_name] = labels
	#     curr_image_names = list(attr_labels.keys())
	#     curr_image_names.sort()
	#     assert self.image_list == curr_image_names, [len(self.image_list), len(curr_image_names)]
	#     self.attr_labels = attr_labels

	def __len__(self):
		if self.gen_pseudolabel:
			return len(self.original_image_list)
		return len(self.image_list)

	def __getitem__(self, index):
		if self.gen_pseudolabel:
			image_name = self.original_image_list[index]
			path = osp.join(self.root, self.original_image_list[index])
		else:
			image_name = self.image_list[index]
			path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		if self.mode == 'val':
			face = self.transform(sample)
			return face, 0

		weak_sample = self.weak_transform(sample)

		if self.gen_pseudolabel:
			return weak_sample, index

		strong_sample = self.strong_transform(sample)

		if self.ssp:
			strong_sample2 = self.strong_transform(sample)
			return weak_sample, strong_sample, strong_sample2

		return weak_sample, strong_sample
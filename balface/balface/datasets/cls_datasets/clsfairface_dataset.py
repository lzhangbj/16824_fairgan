import os
import os.path as osp
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur

# CUDA_VISIBLE_DEVICES=0 python fairface_project.py --outdir=fairface_train --target=../datasets/FairFace/images/train --label-file=../datasets/FairFace/labels/all/fairface_train_all_label.txt --race White  --network=./pretrained/ffhq.pkl

class CLSFairFaceDataset(Dataset):
	ATTRS_NUM = {
		'age': 9,
		'gender': 2,
		'race': 7
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'Latino_Hispanic': 2,
			'East Asian': 3,
			'Southeast Asian': 4,
			'Indian': 5,
			'Middle Eastern': 6
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
			2: 'Latino_Hispanic',
			3: 'East Asian',
			4: 'Southeast Asian',
			5: 'Indian',
			6: 'Middle Eastern'
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

	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFaceDataset, self).__init__()
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

		attr_labels = {}
		self.image_list = []
		self.bbox_labels = {}

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
				labels = []
				for i in range(len(self.attrs)):
					attr = self.attrs[i]
					name = line[i+1]
					labels.append(self.ATTRS_LIST[attr][name])

				labels = np.array(labels).astype(np.int32)
				attr_labels[image_name] = labels
				assert bbox_labels.get(image_name) is not None
				self.bbox_labels[image_name] = bbox_labels[image_name]
				self.image_list.append(image_name)

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
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0.25
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0

		self.transform = transforms.Compose(augmentation)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# crop face with padding from image
		w, h = sample.size
		x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		center_w, center_h = (x0+x1)//2, (y0+y1)//2
		face_w, face_h = x1-x0+1, y1-y0+1
		lateral = max(face_w, face_h)
		lateral = int(lateral * (1+self.pad_ratio))

		x0, x1 = center_w - lateral//2, center_w + lateral//2
		y0, y1 = center_h - lateral//2, center_h + lateral//2

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# face_w, face_h = x1-x0+1, y1-y0+1
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio/2), int(face_h * self.pad_ratio/2)
		# x0, y0, x1, y1 = x0-face_pad_w, y0-face_pad_h, x1+face_pad_w, y1+face_pad_h
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)

		sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels


class CLSFairFace4RacesDataset(Dataset):
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


	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace4RacesDataset, self).__init__()

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

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
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

		return face, labels


class CLSFairFace5RacesDataset(Dataset):
	ATTRS_NUM = {
		'age': 9,
		'gender': 2,
		'race': 5
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'East Asian': 2,
			'Southeast Asian': 3,
			'Indian': 4
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
			3: 'Southeast Asian',
			4: 'Indian'
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


	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace5RacesDataset, self).__init__()

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

				if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
				for i in range(len(self.attrs)):
					attr = self.attrs[i]
					name = line[i+1]
					labels.append(self.ATTRS_LIST[attr][name])

				labels = np.array(labels).astype(np.int32)
				attr_labels[image_name] = labels
				assert bbox_labels.get(image_name) is not None

				self.bbox_labels[image_name] = bbox_labels[image_name]
				self.image_list.append(image_name)

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
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),
				# transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0.25
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0

		self.transform = transforms.Compose(augmentation)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# center_w, center_h = (x0+x1)//2, (y0+y1)//2
		# face_w, face_h = x1-x0+1, y1-y0+1
		# lateral = max(face_w, face_h)
		# lateral = int(lateral * (1+self.pad_ratio))
		#
		# x0, x1 = center_w - lateral//2, center_w + lateral//2
		# y0, y1 = center_h - lateral//2, center_h + lateral//2

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# face_w, face_h = x1-x0+1, y1-y0+1
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio/2), int(face_h * self.pad_ratio/2)
		# x0, y0, x1, y1 = x0-face_pad_w, y0-face_pad_h, x1+face_pad_w, y1+face_pad_h
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)

		# sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels


class CLSFairFace6RacesDataset(Dataset):
	ATTRS_NUM = {
		'age': 9,
		'gender': 2,
		'race': 6
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'East Asian': 2,
			'Southeast Asian': 3,
			'Indian': 4,
			'Middle Eastern': 5
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
			3: 'Southeast Asian',
			4: 'Indian',
			5: 'Middle Eastern'
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


	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace6RacesDataset, self).__init__()

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

				if line[1] in ['Latino_Hispanic',]: continue
				for i in range(len(self.attrs)):
					attr = self.attrs[i]
					name = line[i+1]
					labels.append(self.ATTRS_LIST[attr][name])

				labels = np.array(labels).astype(np.int32)
				attr_labels[image_name] = labels
				assert bbox_labels.get(image_name) is not None

				self.bbox_labels[image_name] = bbox_labels[image_name]
				self.image_list.append(image_name)

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
				transforms.RandomResizedCrop(size, scale=(0.25, 1.)),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0.5
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0

		self.transform = transforms.Compose(augmentation)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# center_w, center_h = (x0+x1)//2, (y0+y1)//2
		# face_w, face_h = x1-x0+1, y1-y0+1
		# lateral = max(face_w, face_h)
		# lateral = int(lateral * (1+self.pad_ratio))
		#
		# x0, x1 = center_w - lateral//2, center_w + lateral//2
		# y0, y1 = center_h - lateral//2, center_h + lateral//2

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# face_w, face_h = x1-x0+1, y1-y0+1
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio/2), int(face_h * self.pad_ratio/2)
		# x0, y0, x1, y1 = x0-face_pad_w, y0-face_pad_h, x1+face_pad_w, y1+face_pad_h
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)

		# sample = sample.crop((x0, y0, x1, y1))

		# crop face with padding from image
		w, h = sample.size
		x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		lateral = max(face_w, face_h)
		lateral = int(lateral * (1 + self.pad_ratio))

		x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		y0, y1 = center_h - lateral // 2, center_h + lateral // 2

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# face_w, face_h = x1-x0+1, y1-y0+1
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio/2), int(face_h * self.pad_ratio/2)
		# x0, y0, x1, y1 = x0-face_pad_w, y0-face_pad_h, x1+face_pad_w, y1+face_pad_h
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)

		sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels


class CLSFairFace7RacesDataset(Dataset):
	ATTRS_NUM = {
		'age': 9,
		'gender': 2,
		'race': 7
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'Latino_Hispanic': 2,
			'East Asian': 3,
			'Southeast Asian': 4,
			'Indian': 5,
			'Middle Eastern': 6
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
			2: 'Latino_Hispanic',
			3: 'East Asian',
			4: 'Southeast Asian',
			5: 'Indian',
			6: 'Middle Eastern'
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

	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace7RacesDataset, self).__init__()

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

				# if line[1] in ['Latino_Hispanic',]: continue
				for i in range(len(self.attrs)):
					attr = self.attrs[i]
					name = line[i+1]
					labels.append(self.ATTRS_LIST[attr][name])

				labels = np.array(labels).astype(np.int32)
				attr_labels[image_name] = labels
				assert bbox_labels.get(image_name) is not None

				self.bbox_labels[image_name] = bbox_labels[image_name]
				self.image_list.append(image_name)

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
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),
				# transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0.25
		else:
			augmentation = [
				transforms.Resize((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
			self.pad_ratio = 0

		self.transform = transforms.Compose(augmentation)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# center_w, center_h = (x0+x1)//2, (y0+y1)//2
		# face_w, face_h = x1-x0+1, y1-y0+1
		# lateral = max(face_w, face_h)
		# lateral = int(lateral * (1+self.pad_ratio))
		#
		# x0, x1 = center_w - lateral//2, center_w + lateral//2
		# y0, y1 = center_h - lateral//2, center_h + lateral//2

		# crop face with padding from image
		# w, h = sample.size
		# x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# face_w, face_h = x1-x0+1, y1-y0+1
		# face_pad_w, face_pad_h = int(face_w * self.pad_ratio/2), int(face_h * self.pad_ratio/2)
		# x0, y0, x1, y1 = x0-face_pad_w, y0-face_pad_h, x1+face_pad_w, y1+face_pad_h
		# x0 = max(0, x0)
		# x1 = min(w, x1)
		# y0 = max(0, y0)
		# y1 = min(h, y1)

		# sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels


class CLSFairFace4RacesVisDataset(Dataset):
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


	def __init__(self, root, face_bbox_txt, face_label_txt, size=128, pad=1.0):
		super(CLSFairFace4RacesVisDataset, self).__init__()

		self.root = root
		self.size = size
		self.pad = pad

		bbox_labels = {}
		with open(face_bbox_txt, 'r') as f:
			for line in f.readlines():
				line = line.strip().split(',')
				image_name = line[0]
				bbox = np.array(list(map(int, line[1:])), dtype=np.int32) # (x0, y0, x1, y1)
				bbox_labels[image_name] = bbox

		attr_labels = {}
		self.image_list = []
		self.bbox_labels = {}

		with open(face_label_txt, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i==0:
					continue
				line = line.strip().split(',')
				image_name = line[0]
				if line[1] == 'Latino_Hispanic': continue

				self.bbox_labels[image_name] = bbox_labels[image_name]
				self.image_list.append(image_name)

		self.image_list.sort()

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# crop face with padding from image
		w, h = sample.size
		x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		lateral = max(face_w, face_h)
		lateral = int(lateral * (1 + self.pad))

		x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		y0, y1 = center_h - lateral // 2, center_h + lateral // 2

		sample = sample.crop((x0, y0, x1, y1))
		sample = sample.resize((self.size, self.size), Image.BILINEAR)
		
		image_name = self.image_list[index]

		return sample, image_name


class CLSFairFace4Races025Dataset(Dataset):
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


	def __init__(self, root, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace4Races025Dataset, self).__init__()

		self.root = root
		self.size = size
		self.ssp = ssp

		attr_labels = {}
		self.image_list = []

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
				self.image_list.append(image_name)

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
		else:
			augmentation = [
				transforms.Resize((size, size)),
				# transforms.CenterCrop((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]

		self.transform = transforms.Compose(augmentation)
		self.mode = mode

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# if self.mode == 'val':
		# 	# crop face with padding from image
		# 	w, h = sample.size
		# 	x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# 	center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		# 	face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		# 	lateral = max(face_w, face_h)
		# 	lateral = int(lateral * (1 + self.pad_ratio))
		#
		# 	x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		# 	y0, y1 = center_h - lateral // 2, center_h + lateral // 2
		#
		# 	sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels, 1.0


class CLSFairFace4Races025augDataset(Dataset):
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

	def __init__(self, root, face_label_txt=None, aug_face_label_txt=None, weights_list=None, aug_face_label_txt_list=[], size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace4Races025augDataset, self).__init__()

		self.root = root
		self.size = size
		self.ssp = ssp

		attr_labels = {}
		self.image_list = []
		
		self.attrs = ['race','gender', 'age']

		if face_label_txt:
			with open(face_label_txt, 'r') as f:
				for i, line in enumerate(f.readlines()):
					if i==0:
						continue
					line = line.strip().split(',')
					image_name = line[0]
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
					self.image_list.append(image_name)
				
		self.num_split = [len(self.image_list), ]
		if len(aug_face_label_txt_list)==0: aug_face_label_txt_list.append(aug_face_label_txt)
		for aug_face_label_txt in aug_face_label_txt_list:
			with open(aug_face_label_txt, 'r') as f:
				for i, line in enumerate(f.readlines()):
					if i==0:
						continue
					line = line.strip().split(',')
					image_name = line[0]
					labels = []
	
					if line[1] == 'Middle Eastern': line[1] = 'White'
					if line[1] == 'Southeast Asian': line[1] = 'East Asian'
					if line[1] in ['Latino_Hispanic', 'Middle Eastern'] : continue
					# if line[1] == 'White': continue
					for i in range(len(self.attrs)):
						if i == 0:
							attr = self.attrs[i]
							name = line[i+1]
							labels.append(self.ATTRS_LIST[attr][name])
						else:
							labels.append(0)
	
					labels = np.array(labels).astype(np.int32)
					attr_labels[image_name] = labels
					self.image_list.append(image_name)
			self.num_split.append(len(self.image_list))
		
		self.aug_weight_list = weights_list if weights_list is not None else [1.0, ]*len(aug_face_label_txt_list)
		assert len(self.aug_weight_list) == len(aug_face_label_txt_list)
		self.aug_weight_list = [1.0, ] + self.aug_weight_list

		# self.image_list.sort()
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
		else:
			augmentation = [
				transforms.Resize((size, size)),
				# transforms.CenterCrop((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]

		self.transform = transforms.Compose(augmentation)
		self.mode = mode

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		weight_index = 0
		while weight_index < len(self.num_split) and index < self.num_split[weight_index]:
			weight_index += 1
		weight_index -= 1
		weight = self.aug_weight_list[weight_index]
		path = osp.join(self.root, self.image_list[index])

		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')

		# if self.mode == 'val':
		# 	# crop face with padding from image
		# 	w, h = sample.size
		# 	x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# 	center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		# 	face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		# 	lateral = max(face_w, face_h)
		# 	lateral = int(lateral * (1 + self.pad_ratio))
		#
		# 	x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		# 	y0, y1 = center_h - lateral // 2, center_h + lateral // 2
		#
		# 	sample = sample.crop((x0, y0, x1, y1))

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels, weight


class CLSFairFace4Races025augDenoiseDataset(Dataset):
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
	
	def __init__(self, root, face_label_txt=None, return_index=False, aug_face_label_txt=None,  weights_list=None,
	             aug_face_label_txt_list=[], size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace4Races025augDenoiseDataset, self).__init__()
		
		self.root = root
		self.size = size
		self.ssp = ssp
		self.return_index = return_index
		
		attr_labels = {}
		self.image_list = []
		
		self.attrs = ['race', 'gender', 'age']
		
		self.out_labels = []
		
		if face_label_txt:
			with open(face_label_txt, 'r') as f:
				for i, line in enumerate(f.readlines()):
					if i == 0:
						continue
					line = line.strip().split(',')
					image_name = line[0]
					labels = []
					
					if line[1] == 'Middle Eastern': line[1] = 'White'
					if line[1] == 'Southeast Asian': line[1] = 'East Asian'
					if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
					for i in range(len(self.attrs)):
						attr = self.attrs[i]
						name = line[i + 1]
						labels.append(self.ATTRS_LIST[attr][name])
					
					labels = np.array(labels).astype(np.int32)
					attr_labels[image_name] = labels
					self.image_list.append(image_name)
					self.out_labels.append(int(labels[0]))
		
		self.num_split = [len(self.image_list), ]
		if len(aug_face_label_txt_list) == 0: aug_face_label_txt_list.append(aug_face_label_txt)
		
		self.aug_weight_list = weights_list if weights_list is not None else [1.0, ] * len(aug_face_label_txt_list)
		assert len(self.aug_weight_list) == len(aug_face_label_txt_list)
		
		for jj, aug_face_label_txt in enumerate(aug_face_label_txt_list):
			with open(aug_face_label_txt, 'r') as f:
				for i, line in enumerate(f.readlines()):
					if i == 0:
						continue
					line = line.strip().split(',')
					image_name = line[0]
					labels = []
					
					if line[1] == 'Middle Eastern': line[1] = 'White'
					if line[1] == 'Southeast Asian': line[1] = 'East Asian'
					if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
					for i in range(len(self.attrs)):
						if i == 0:
							attr = self.attrs[i]
							name = line[i + 1]
							labels.append(self.ATTRS_LIST[attr][name])
						else:
							labels.append(0)
					
					labels = np.array(labels).astype(np.int32)
					attr_labels[image_name] = labels
					self.image_list.append(image_name)
					self.out_labels.append(int(labels[0]))
					
			self.num_split.append(len(self.image_list))
			
		self.aug_weight_list = [1.0, ] + self.aug_weight_list

		# self.image_list.sort()
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
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),  # ratio=(1.0, 1.0)
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
		else:
			augmentation = [
				transforms.Resize((size, size)),
				# transforms.CenterCrop((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
		
		self.transform = transforms.Compose(augmentation)
		self.mode = mode
	
	def __len__(self):
		return len(self.image_list)
	
	def get_labels(self):
		return np.array(self.out_labels).astype(np.int32)
	
	def __getitem__(self, index):
		weight_index = 0
		while weight_index < len(self.num_split) and index < self.num_split[weight_index]:
			weight_index += 1
		weight_index -= 1
		weight = self.aug_weight_list[weight_index]
		path = osp.join(self.root, self.image_list[index])
		
		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')
		
		# if self.mode == 'val':
		# 	# crop face with padding from image
		# 	w, h = sample.size
		# 	x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# 	center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		# 	face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		# 	lateral = max(face_w, face_h)
		# 	lateral = int(lateral * (1 + self.pad_ratio))
		#
		# 	x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		# 	y0, y1 = center_h - lateral // 2, center_h + lateral // 2
		#
		# 	sample = sample.crop((x0, y0, x1, y1))
		
		face = self.transform(sample)
		
		labels = self.attr_labels[self.image_list[index]]
		
		labels = torch.from_numpy(labels).long()
		
		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels
		if not self.return_index:
			return face, labels, weight
		return face, labels, weight, index


class CLSFairFace4Races025ratioaugDataset(Dataset):
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
	
	def __init__(self, root, face_label_txt, aug_face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFace4Races025ratioaugDataset, self).__init__()
		
		self.root = root
		self.size = size
		self.ssp = ssp
		
		attr_labels = {}
		self.image_list = []
		
		with open(face_label_txt, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i == 0:
					attrs = line.strip().split(',')[1:]
					for attr in attrs:
						assert attr in self.ATTRS_NUM
					self.attrs = attrs
					continue
				line = line.strip().split(',')
				image_name = line[0]
				labels = []
				
				if line[1] == 'Middle Eastern': line[1] = 'White'
				if line[1] == 'Southeast Asian': line[1] = 'East Asian'
				if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
				for i in range(len(self.attrs)):
					attr = self.attrs[i]
					name = line[i + 1]
					labels.append(self.ATTRS_LIST[attr][name])
				label = labels[0]
				
				labels = np.eye(4)[label].astype(np.float32)
				attr_labels[image_name] = labels
				self.image_list.append(image_name)
		
		
		with open(aug_face_label_txt, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i == 0:
					continue
				line = line.strip().split(',')
				image_name = line[0]
				labels = []
				str_list = line[1:5]
				label = list(map(float, str_list))
				# if line[1] == 'Middle Eastern': line[1] = 'White'
				# if line[1] == 'Southeast Asian': line[1] = 'East Asian'
				# if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
				# for i in range(len(self.attrs)):
				# 	attr = self.attrs[i]
				# 	name = line[i + 1]
				# 	labels.append(self.ATTRS_LIST[attr][name])
				
				labels = np.array(label).astype(np.float32)
				attr_labels[image_name] = labels
				self.image_list.append(image_name)
		
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
				transforms.RandomResizedCrop(size, scale=(0.8, 1.)),  # ratio=(1.0, 1.0)
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
		else:
			augmentation = [
				transforms.Resize((size, size)),
				# transforms.CenterCrop((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std)
			]
		
		self.transform = transforms.Compose(augmentation)
		self.mode = mode
	
	def __len__(self):
		return len(self.image_list)
	
	def __getitem__(self, index):
		path = osp.join(self.root, self.image_list[index])
		
		with open(path, 'rb') as f:
			sample = Image.open(f).convert('RGB')
		
		# if self.mode == 'val':
		# 	# crop face with padding from image
		# 	w, h = sample.size
		# 	x0, y0, x1, y1 = self.bbox_labels[self.image_list[index]]
		# 	center_w, center_h = (x0 + x1) // 2, (y0 + y1) // 2
		# 	face_w, face_h = x1 - x0 + 1, y1 - y0 + 1
		# 	lateral = max(face_w, face_h)
		# 	lateral = int(lateral * (1 + self.pad_ratio))
		#
		# 	x0, x1 = center_w - lateral // 2, center_w + lateral // 2
		# 	y0, y1 = center_h - lateral // 2, center_h + lateral // 2
		#
		# 	sample = sample.crop((x0, y0, x1, y1))
		
		face = self.transform(sample)
		
		labels = self.attr_labels[self.image_list[index]]
		
		labels = torch.from_numpy(labels[None, ...]).float()
		
		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels
		
		return face, labels


if __name__ == '__main__':
	dataset = CLSFairFace4RacesVisDataset(
		root="./datasets/FairFace/images",
        face_bbox_txt="./datasets/FairFace/bbox/fairface_val_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/all/fairface_val_all_label.txt",
        size=256,
		pad=1.0)
	save_dir = '/opt/tiger/balface/datasets/FairFace/images_pad-1_4races'
	os.makedirs(save_dir, exist_ok=True)
	from tqdm import tqdm
	for pil_image, image_name in tqdm(dataset):
		save_image_name = os.path.join(save_dir, image_name)
		os.makedirs(os.path.dirname(save_image_name), exist_ok=True)
		pil_image.save(save_image_name)
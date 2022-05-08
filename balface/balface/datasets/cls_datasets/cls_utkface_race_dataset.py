import os
import os.path as osp
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur


class CLSUTKFaceRacesDataset(Dataset):
	ATTRS_NUM = {
		'age': 8,
		'gender': 2,
		'race': 4
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'Asian': 2,
			'Indian': 3
		},
		'gender': {
			'male': 0,
			'female': 1
		},
		'age': {
			'0-10': 0,
			'11-20': 1,
			'21-30': 2,
			'31-40': 3,
			'41-50': 4,
			'51-60': 5,
			'61-70': 6,
			'70+': 7
		}
	}

	ATTRS_REV_INDEX = {
		'race': {
			0: 'White',
			1: 'Black',
			2: 'Asian',
			3: 'Indian'
		},
		'gender': {
			0: 'male',
			1: 'female'
		},
		'age': {
			0: '0-10',
			1: '11-20',
			2: '21-30',
			3: '31-40',
			4: '41-50',
			5: '51-60',
			6: '61-70',
			7: '70+'
		}
	}


	def __init__(self, root, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSUTKFaceRacesDataset, self).__init__()

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
				labels = []
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


class CLSUTKFaceRacesAugDataset(Dataset):
	ATTRS_NUM = {
		'age': 8,
		'gender': 2,
		'race': 4
	}

	ATTRS_LIST = {
		'race': {
			'White': 0,
			'Black': 1,
			'Asian': 2,
			'Indian': 3
		},
		'gender': {
			'male': 0,
			'female': 1
		},
		'age': {
			'0-10': 0,
			'11-20': 1,
			'21-30': 2,
			'31-40': 3,
			'41-50': 4,
			'51-60': 5,
			'61-70': 6,
			'70+': 7
		}
	}

	ATTRS_REV_INDEX = {
		'race': {
			0: 'White',
			1: 'Black',
			2: 'Asian',
			3: 'Indian'
		},
		'gender': {
			0: 'male',
			1: 'female'
		},
		'age': {
			0: '0-10',
			1: '11-20',
			2: '21-30',
			3: '31-40',
			4: '41-50',
			5: '51-60',
			6: '61-70',
			7: '70+'
		}
	}

	def __init__(self, root, face_label_txt=None, aug_face_label_txt=None, weights_list=None, aug_face_label_txt_list=[], size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSUTKFaceRacesAugDataset, self).__init__()

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
					
					name = line[1]
					labels = [self.ATTRS_LIST['race'][name], 0, 0]
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


if __name__ == '__main__':
	pass
	# dataset = CLSFairFace4RacesVisDataset(
	# 	root="./datasets/FairFace/images",
    #     face_bbox_txt="./datasets/FairFace/bbox/fairface_val_bbox.txt",
    #     face_label_txt="./datasets/FairFace/labels/all/fairface_val_all_label.txt",
    #     size=256,
	# 	pad=1.0)
	# save_dir = '/opt/tiger/balface/datasets/FairFace/images_pad-1_4races'
	# os.makedirs(save_dir, exist_ok=True)
	# from tqdm import tqdm
	# for pil_image, image_name in tqdm(dataset):
	# 	save_image_name = os.path.join(save_dir, image_name)
	# 	os.makedirs(os.path.dirname(save_image_name), exist_ok=True)
	# 	pil_image.save(save_image_name)
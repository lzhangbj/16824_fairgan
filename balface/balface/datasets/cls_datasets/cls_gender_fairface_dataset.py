import os
import os.path as osp
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur

class CLSFairFaceGender025Dataset(Dataset):
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

	def __init__(self, root, face_label_txt, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFaceGender025Dataset, self).__init__()

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
					self.attrs = attrs # (gender, race, age)
					continue
				line = line.strip().split(',')
				image_name = line[0]
				labels = []

				if line[3] in ['0-2', '3-9']: continue
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

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]][[1,0,2]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels, 1.0


class CLSFairFaceGender025augDataset(Dataset):
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

	def __init__(self, root, face_label_txt=None, aug_face_label_txt=None, weights_list=None, aug_face_label_txt_list=[], size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSFairFaceGender025augDataset, self).__init__()

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
					
					if line[3] in ['0-2', '3-9']: continue
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
	
					# if line[3] in ['0-2', '3-9']: continue
					name = line[1]
					if name == 'Male': continue
					labels = [0, self.ATTRS_LIST['gender'][name], 0]
	
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

		face = self.transform(sample)

		labels = self.attr_labels[self.image_list[index]][[1,0,2]]

		labels = torch.from_numpy(labels).long()

		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels

		return face, labels, weight


if __name__ == '__main__':
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
	pass
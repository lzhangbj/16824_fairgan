import os
import os.path as osp
import random
import numpy as np
import torch
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur


class CLSGANFairFace4Races025Dataset(Dataset):
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
	
	def __init__(self, root, face_label_txt, gan_dir, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSGANFairFace4Races025Dataset, self).__init__()
		
		self.root = root
		self.size = size
		self.ssp = ssp
		self.gan_dir = gan_dir
		
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
		
		self.reinit_gan_images()
	
	def reinit_gan_images(self):
		gan_dir = self.gan_dir
		self.gan_image_list = []
		self.gan_attr_labels = []
		for file in glob(f"{self.root}/{gan_dir}/**/*.jpg", recursive=True):
			file_split = file.split('/')[-2:]
			cls_str = file_split[0]
			file_name = file_split[1]
			image_name = f"{gan_dir}/{cls_str}/{file_name}"
			self.gan_image_list.append(image_name)
			label = np.array([self.ATTRS_LIST['race'][cls_str],]).astype(np.int32)
			self.gan_attr_labels.append(label)
		print(f"dataset image length {len(self.image_list)}, gan image length {len(self.gan_image_list)}")
		
			
	def __len__(self):
		return len(self.image_list) + len(self.gan_image_list)
	
	def __getitem__(self, index):
		if index < len(self.image_list):
			path = osp.join(self.root, self.image_list[index])
			labels = self.attr_labels[self.image_list[index]]
			labels = torch.from_numpy(labels).long()
		else:
			gan_index = index-len(self.image_list)
			path = osp.join(self.root, self.gan_image_list[gan_index])
			labels = torch.from_numpy(self.gan_attr_labels[gan_index]).long()
			
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
		
		if self.ssp:
			face2 = self.transform(sample)
			return face, face2, labels
		
		return face, labels


class CLSLatentFairFace4RacesDataset(Dataset):
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


	def __init__(self, root, face_bbox_txt, face_label_txt, face_latent_npy, size=128, mode='train', ssp=False, *args, **kwargs):
		super(CLSLatentFairFace4RacesDataset, self).__init__()

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

		self.latent_list = []
		latent_dict = np.load(face_latent_npy, allow_pickle=True)
		for image_name, latent_np in latent_dict.items():
			self.latent_list[image_name].append(latent_np)

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

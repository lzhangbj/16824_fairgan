import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from balface.models import build_model

cls_list = ['White', 'Black', 'East Asian', 'Indian']
cls_mapping = {
	'White': 'White',
	'Middle Eastern': 'White',
	'Black': 'Black',
	'East Asian': 'East Asian',
	'Southeast Asian': 'East Asian',
	'Indian': 'Indian',
	'Asian': 'East Asian'
}


class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--image-dir', type=str, help='Path to experiment output directory')
	parser.add_argument('--label-file', type=str, default='read images from label-file')
	parser.add_argument('--save-name', type=str, help='Path to experiment output directory, embeddings and label file')
	parser.add_argument('--ckpt', type=str, help='Path to experiment output directory')
	args = parser.parse_args()
	return args


def read_label_file(file):
	labels = {}
	with open(file, 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i == 0: continue
			line = line.strip().split(',')
			image_name = line[0]
			race = line[1]
			if race == 'Latino_Hispanic': continue
			race = cls_mapping[race]
			label = cls_list.index(race)
			labels[image_name] = label
	return labels


class DataSet(Dataset):
	def __init__(self, root, image_paths, transform):
		super(DataSet, self).__init__()
		self.root = root
		self.image_paths = image_paths
		self.transform = transform
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, item):
		image = Image.open(osp.join(self.root, self.image_paths[item]))
		image_tensor = self.transform(image)
		return self.image_paths[item], image_tensor


def main():
	args = parse_args()
	save_name = args.save_name
	
	labels_dict = read_label_file(args.label_file)
	image_paths = list(labels_dict.keys())
	image_paths.sort()
	
	cfg_model = {
		'name': 'cls_recognizer',
		'backbone': Bunch({
			'name': "torch_resnet34"
		}),
		'head': Bunch({
			'name': "multiclassifier_head",
			'n_classes': [4, ],
			'input_dim': 512,
			'hidden_dims': [],
			'loss': 'ce',
			'norm_weights': False
		}),
		'use_weight': False
	}
	cfg_model = Bunch(cfg_model)
	
	net = build_model(cfg_model)
	save_content = torch.load(args.ckpt, map_location='cpu')
	statedict = save_content['state_dict']
	net.load_state_dict(statedict)
	net = net.cuda()
	net.eval()
	
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize(mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
		                     std=np.array([0.229, 0.224, 0.225], dtype=np.float32))
	])
	
	embedding_dict = {}
	label_dict = {}
	
	dataset = DataSet(args.image_dir, image_paths, transform)
	data_loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False, drop_last=False)
	
	with torch.no_grad():
		device = f'cuda:0'
		for image_paths, image_tensor in tqdm(data_loader):
			image_tensor = image_tensor.to(device)
			embeddings, probs, scores = net.produce_importance(image_tensor, method='v2')
			
			pred_labels = torch.argmax(probs, dim=1)
			
			for i in range(len(image_paths)):
				image_path = image_paths[i]
				label = pred_labels[i]
				race = cls_list[label.cpu().numpy().item()]
				embedding = embeddings[i].cpu().numpy()
				embedding_dict[image_path] = embedding
				label_dict[image_path] = race
	
	np.save(f"{save_name}_embeddings", embedding_dict)
	with open(f'{save_name}_labels.txt', 'w') as f:
		f.writelines("image_name,race")
		for image_name, race in label_dict.items():
			f.writelines(f"\n{image_name},{race}")


if __name__ == '__main__':
	main()



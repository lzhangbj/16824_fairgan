import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
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
	'Indian': 'Indian'
}

class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--image_dir', type=str, help='Path to experiment output directory')
	parser.add_argument('--label-file', type=str, default='')
	parser.add_argument('--save-name', type=str, help='Path to experiment output directory')
	parser.add_argument('--ckpt', type=str, help='Path to experiment output directory')
	args = parser.parse_args()
	return args

def read_label_file(file):
	labels = {}
	with open(file, 'r') as f:
		for i,line in enumerate(f.readlines()):
			if i==0: continue
			line = line.strip().split(',')
			image_name = line[0]
			race = line[1]
			if race == 'Latino_Hispanic': continue
			race = cls_mapping[race]
			label = cls_list.index(race)
			labels[image_name] = label
	return labels


class DataSet(Dataset):
	def __init__(self, image_paths, transform):
		super(DataSet, self).__init__()
		self.image_paths = image_paths
		self.transform = transform
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, item):
		image = Image.open(self.image_paths[item])
		image_tensor = self.transform(image)
		return self.image_paths[item], image_tensor


def main():
	args = parse_args()
	image_paths = glob(f"{args.image_dir}/**/*.jpg", recursive=True)
	image_paths = [image_path for image_path in image_paths if 'sorted' not in image_path]
	image_paths.sort()
	save_name = args.save_name
	
	if args.label_file:
		labels_dict = read_label_file(args.label_file)
	
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
				# transforms.CenterCrop((size, size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
									 std=np.array([0.229, 0.224, 0.225], dtype=np.float32))
			])
	
	embedding_dict = {}
	score_dict = {}
	acc_dict = {
		'true_label': {},
		'aug_label': {}
	}
	dataset = DataSet(image_paths, transform)
	data_loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False, drop_last=False)
	
	with torch.no_grad():
		device = f'cuda:0'
		for image_paths, image_tensor in tqdm(data_loader):
			image_tensor = image_tensor.to(device)
			embeddings, probs, scores = net.produce_importance(image_tensor, method='v2')
			
			for i in range(len(image_paths)):
				image_path = image_paths[i]
				# get label
				if args.label_file:
					try:
						image_name = image_path.split('/')[-3:]
						image_name = '/'.join(image_name)
						pred_label = labels_dict[image_name]
						true_label = torch.argmax(probs, dim=1).detach().cpu().numpy()[i].item()
						acc_dict['true_label'][image_path] = true_label
						acc_dict['aug_label'][image_path] = pred_label
					except:
						continue
				
				score = scores[i].cpu().numpy().item()
				embedding = embeddings[i].cpu().numpy()
				# image_name = image_path.split('/')[-1]
				score_dict[image_path] = score
				embedding_dict[image_path] = embedding
		
		
		# for image_path in tqdm(image_paths):
		# 	image = Image.open(image_path)
		# 	image_tensor = transform(image).unsqueeze(0).cuda()
		#
		# 	embeddings, probs, scores = net.produce_importance(image_tensor, method='v2')
		#
		# 	# get label
		# 	if args.label_file:
		# 		try:
		# 			image_name = image_path.split('/')[-2:]
		# 			image_name = '/'.join(image_name)
		# 			pred_label = labels_dict[image_name]
		# 			true_label = torch.argmax(probs, dim=1).detach().cpu().numpy()[0].item()
		# 			acc_dict['true_label'][image_path] = true_label
		# 			acc_dict['aug_label'][image_path] = pred_label
		# 		except:
		# 			continue
		#
		# 	score = scores[0].cpu().numpy().item()
		# 	embedding = embeddings[0].cpu().numpy()
		# 	# image_name = image_path.split('/')[-1]
		# 	score_dict[image_path] = score
		# 	embedding_dict[image_path] = embedding
			

	
	np.save(f"{save_name}_scores", score_dict)
	np.save(f"{save_name}_embeddings", embedding_dict)
	if args.label_file:
		np.save(f"{save_name}_acc_labels", acc_dict)
	
if __name__ =='__main__':
	main()
	
	
	
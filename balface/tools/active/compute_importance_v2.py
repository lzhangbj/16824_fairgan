import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import copy
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from glob import glob
from PIL import Image
import multiprocessing

from balface.models import build_model

torch.multiprocessing.set_start_method('spawn', force=True)


class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--image_dir', type=str, help='Path to experiment output directory')
	parser.add_argument('--save_dir', type=str, help='Path to experiment output directory')
	parser.add_argument('--ckpt', type=str, help='Path to experiment output directory')
	args = parser.parse_args()
	return args


def multi_run_wrapper(args):
	return inference(*args)

def inference(image_paths, transform, net, device='cuda:0'):
	cls_score_dict = {}
	all_scores = []
	for image_path in tqdm(image_paths):
		image = Image.open(image_path)
		image_tensor = transform(image).unsqueeze(0).to(device)
		
		_, _, scores = net.produce_importance(image_tensor, method='v2')
		score = scores[0].cpu().numpy().item()
		
		image_name = image_path.split('/')[-1]
		cls_score_dict[image_name] = score
		all_scores.append(score)
	
	return cls_score_dict, all_scores

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
	cls_list = ['White', 'Black', 'East Asian', 'Indian']
	
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
	
	score_dict = {}
	all_scores = []
	
	params = []
	for i, cls in enumerate(cls_list[1:]):
		image_paths = glob(f"{args.image_dir}/{cls}/*.jpg")
		image_paths.sort()
		
		dataset = DataSet(image_paths, transform)
		data_loader = DataLoader(dataset, batch_size=20, num_workers=4, shuffle=False, drop_last=False)

		save_dir = os.path.join(args.save_dir, cls)
		os.makedirs(save_dir, exist_ok=True)
		cls_score_dict = {}
		device = f'cuda:0'
		for image_paths, image_tensor in tqdm(data_loader):
			image_tensor = image_tensor.to(device)
			
			_, _, scores = net.produce_importance(image_tensor, method='v2')
			for i in range(len(scores)):
				score = scores[i].cpu().numpy().item()
				
				image_name = image_paths[i].split('/')[-1]
				cls_score_dict[image_name] = score
				all_scores.append(score)
		score_dict[cls] = cls_score_dict
	# return cls_score_dict, all_scores
		
	# 	params.append([image_paths, transform, copy.deepcopy(net).to(f'cuda:{i}'), f'cuda:{i}'])
	#
	# results = multiprocessing.Pool(4).map(multi_run_wrapper, params)
	#
	# for cls_score_dict, cls_all_scores in results:
	# 	score_dict.update(cls_score_dict)
	# 	all_scores += cls_all_scores
		
	np.save(f'{args.save_dir}/scores', score_dict)
		
	all_scores = np.array(all_scores)
	min_score = all_scores.min()
	max_score = all_scores.max()

	couple_save_dir = os.path.join(args.save_dir, 'sorted_images')
	for cls in cls_list[1:]:
		cls_couple_save_dir = os.path.join(couple_save_dir, cls)
		os.makedirs(cls_couple_save_dir, exist_ok=True)
		
		image_paths = glob(f"{args.image_dir}/{cls}/*.jpg")
		for image_path in tqdm(image_paths):
			image_name = image_path.split('/')[-1]
			score = score_dict[cls][image_name]
			score = (score - min_score) / (max_score - min_score)
			save_path = os.path.join(cls_couple_save_dir, f"{score:.5f}_{image_name}")
			shutil.copyfile(image_path, save_path)
			

	
if __name__ =='__main__':
	main()
		
	
	
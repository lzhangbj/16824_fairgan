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
	parser.add_argument('--ckpt', type=str, help='Path to experiment output directory')
	parser.add_argument('--ckpt2', type=str, help='Path to experiment output directory')
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
	
	if args.ckpt2:
		net2 = build_model(cfg_model)
		save_content = torch.load(args.ckpt2, map_location='cpu')
		statedict = save_content['state_dict']
		net2.load_state_dict(statedict)
		net2 = net2.cuda()
		net2.eval()
	
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		# transforms.CenterCrop((size, size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
							 std=np.array([0.229, 0.224, 0.225], dtype=np.float32))
	])
	
	from sklearn.metrics import confusion_matrix
	
	gts = []
	preds = []
	gt_dict = {}
	baseline_dict = {}
	for i, cls in enumerate(cls_list[1:]):
		image_paths = glob(f"{args.image_dir}/{cls}/*.jpg")
		image_paths.sort()
		
		cls_gts = []
		
		dataset = DataSet(image_paths, transform)
		data_loader = DataLoader(dataset, batch_size=20, num_workers=4, shuffle=False, drop_last=False)

		device = f'cuda:0'
		
		cls_preds = []
		for image_paths, image_tensor in tqdm(data_loader):
			image_tensor = image_tensor.to(device)
			
			_, probs, _ = net.produce_importance(image_tensor, method='v2')
			cls_gts.append(np.argmax(probs.cpu().numpy(), axis=1))
			
			if args.ckpt2:
				_, probs2, _ = net2.produce_importance(image_tensor, method='v2')
				cls_preds.append(np.argmax(probs2.cpu().numpy(), axis=1))
			
			for i, image_path in enumerate(image_paths):
				gi = np.argmax(probs.cpu().numpy(), axis=1)[i]
				gt_dict[image_path.split('/')[-1]] = gi.item()
				if args.ckpt2:
					gi = np.argmax(probs2.cpu().numpy(), axis=1)[i]
					baseline_dict[image_path.split('/')[-1]] = gi.item()
		
		cls_gts = np.concatenate(cls_gts).astype(np.int32)
		if len(cls_preds) == 0:
			cls_preds = (np.ones_like(cls_gts) * cls_list.index(cls)).astype(np.int32)
		else:
			cls_preds = np.concatenate(cls_preds).astype(np.int32)
		gts.append(cls_gts)
		preds.append(cls_preds)
	print('computing confusion matrix ... ')
	gts = np.concatenate(gts)
	preds = np.concatenate(preds)
	conf_matrix = confusion_matrix(gts, preds)
	print(conf_matrix)
	
	np.save(f'{args.image_dir}/gt', gt_dict)
	
	if args.ckpt2:
		np.save(f'{args.image_dir}/baseline_pred', baseline_dict)
	

	
if __name__ =='__main__':
	main()
		
	
	
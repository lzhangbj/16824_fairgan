import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
import os.path as osp
import random
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from glob import glob
from PIL import Image


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--src-root', type=str, help='Path to experiment output directory')
	parser.add_argument('--dest-root', type=str, help='Path to experiment output directory')
	parser.add_argument('--dest-label-file', type=str, help='Path to experiment output directory')
	parser.add_argument('--importance-file', type=str, help='Path to experiment output directory')
	parser.add_argument('--sample-method', default='V1', type=str, help='Path to experiment output directory')
	parser.add_argument('--sample-num', type=int, help='Path to experiment output directory')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	cls_list = ['White', 'Black', 'East Asian', 'Indian']
	
	importance_dict = np.load(args.importance_file, allow_pickle=True).item()
	
	selected_image_dict = {}
	
	random.seed(123)
	np.random.seed(123)
	
	for cls in cls_list[1:]:
		print(f"producing {args.sample_num} aug images for class {cls}")
		selected_image_dict[cls] = []
		
		dest_image_root = osp.join(args.dest_root, cls)
		# if osp.exists(dest_image_root)
		os.makedirs(dest_image_root, exist_ok=True)
		dest_image_prefix = osp.join(osp.basename(args.dest_root), cls)
		
		cls_score_dict = importance_dict[cls]
		cls_image_names = []
		cls_importances = []
		for image_name, score in cls_score_dict.items():
			cls_image_names.append(image_name)
			cls_importances.append(score)
		cls_importance = np.array(cls_importances)
		# sample v1 --- importance sample
		sorted_idx = np.argsort(-cls_importance)
		selected_idx = sorted_idx[:args.sample_num]
		# sample v2 --- mean importance sample
		# cls_importance_mean = np.mean(cls_importance)
		# dist = np.abs(cls_importance - cls_importance_mean)
		# sorted_idx = np.argsort(dist)
		# selected_idx = sorted_idx[:args.sample_num]
		# sample v3 --- random sample 540
		# selected_idx = np.arange(len(cls_importance)).astype(np.int32)
		# np.random.shuffle(selected_idx)
		# selected_idx = selected_idx[:args.sample_num]
		
		for idx in selected_idx:
			selected_image_name = cls_image_names[idx]
			# append to label file
			selected_image_dict[cls].append(osp.join(dest_image_prefix, selected_image_name))
			# copy file to dest
			src_image_path = osp.join(osp.join(osp.join(args.src_root), cls), selected_image_name)
			dest_image_path = osp.join(dest_image_root, selected_image_name)
			assert osp.exists(src_image_path)
			shutil.move(src_image_path, dest_image_path)
	
	with open(args.dest_label_file, 'w') as f:
		f.writelines('image_name,race')
		for cls in cls_list[1:]:
			for image_name in selected_image_dict[cls]:
				f.writelines(f"\n{image_name},{cls}")

if __name__ == '__main__':
	main()



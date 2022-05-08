import argparse
import logging
import os
import os.path as osp
import time
import copy
import warnings
from glob import glob
from tqdm import tqdm
from PIL import Image
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms

from mmcv import Config, DictAction

from balface.models import build_model
from balface.datasets import build_dataset
from balface.utils import AverageMeter, get_root_logger,set_random_seed


def parse_args():
	parser = argparse.ArgumentParser(description='Train a detector')
	parser.add_argument('config', help='train config file path')
	parser.add_argument('--work-dir', help='the dir to save logs and models')
	parser.add_argument(
		'--resume-from', help='the checkpoint file to resume from')
	parser.add_argument(
		'--no-validate',
		action='store_true',
		help='whether not to evaluate the checkpoint during training')
	group_gpus = parser.add_mutually_exclusive_group()
	group_gpus.add_argument(
		'--gpus',
		type=int,
		nargs='+',
		help='ids of gpus to use '
			 '(only applicable to non-distributed training)')
	parser.add_argument('--seed', type=int, default=123, help='random seed')
	parser.add_argument(
		'--deterministic',
		action='store_true',
		help='whether to set deterministic options for CUDNN backend.')
	parser.add_argument(
		'--cfg-options',
		nargs='+',
		action=DictAction,
		help='override some settings in the used config, the key-value pair '
		'in xxx=yyy format will be merged into config file. If the value to '
		'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
		'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
		'Note that the quotation marks are necessary and that no white space '
		'is allowed.')

	parser.add_argument('--test_data_root', type=str, help='the dir to save logs and models')
	parser.add_argument('--load_from', type=str, help='the dir to save logs and models')
	parser.add_argument('--save_name', type=str, default="embeddings_importance", help='the dir to save logs and models')
	parser.add_argument('--method', type=str, default="v1", help='the dir to save logs and models')
	parser.add_argument('--val', action='store_true', help='the dir to save logs and models')

	args = parser.parse_args()

	return args


def main():
	args = parse_args()

	cfg = Config.fromfile(args.config)
	if args.cfg_options is not None:
		cfg.merge_from_dict(args.cfg_options)
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	
	# work_dir is determined in this priority: CLI > segment in file > filename
	if args.work_dir is not None:
		# update configs according to CLI args if args.work_dir is not None
		cfg.work_dir = args.work_dir
	elif cfg.get('work_dir', None) is None:
		# use config filename as default work_dir if cfg.work_dir is None
		cfg.work_dir = osp.join('./work_dirs',
		                        osp.splitext(osp.basename(args.config))[0])
	os.makedirs(cfg.work_dir, exist_ok=True)

	model = build_model(cfg.model)
	test_data_root = args.test_data_root
	# images = glob(f"{test_data_root}/*.jpg")
	# images.sort()

	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		# transforms.CenterCrop((size, size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std)
	])

	save_content = torch.load(args.load_from, map_location='cpu')
	statedict = save_content['state_dict']
	model.load_state_dict(statedict)
	model = model.cuda()
	model.eval()
	
	if args.val:
		label_file = cfg.data.val.face_label_txt
	else:
		label_file = cfg.data.train.face_label_txt
	label_dict = {}
	cls_list = ['White', 'Black', 'East Asian', 'Indian']
	with open(label_file, 'r') as f:
		for i,line in enumerate(f.readlines()):
			if i==0: continue
			line = line.strip().split(',')
			image_name = line[0]
		
			if line[1] == 'Middle Eastern': line[1] = 'White'
			if line[1] == 'Southeast Asian': line[1] = 'East Asian'
			if line[1] in ['Latino_Hispanic', 'Middle Eastern']: continue
			cls_ind = cls_list.index(line[1])
			label_dict[image_name] = cls_ind
			
	importance_dict = {
		'importance': {},
		'embeddings': {}
	}
	
	acc_list = []
	for image_name, tgt in tqdm(label_dict.items()):
		image_path = os.path.join(args.test_data_root, image_name)
		image_pil = Image.open(image_path)
		# print(image_pil.size)
		image_tensor = transform(image_pil)
		image_tensor = image_tensor.cuda().unsqueeze(0)
		tgt_tensor = torch.tensor(tgt).view(1).cuda()
		# print(image_tensor, tgt_tensor)
		with torch.no_grad():
			embeddings, probs, importance = model.produce_importance(image_tensor, tgt_tensor, method=args.method)
			predict = torch.argmax(probs, dim=1)[0]
			is_correct = (predict == tgt_tensor[0]).cpu().numpy()
			acc_list.append(is_correct)
		embeddings = embeddings[0].cpu().numpy()
		importance = importance[0].cpu().numpy()
		
		importance_dict['importance'][image_name] = importance
		importance_dict['embeddings'][image_name] = embeddings
	
	acc_list = np.array(acc_list).astype(np.float32)
	print("acc is ", acc_list.mean())
	
	np.save(f"{cfg.work_dir}/{args.save_name}", importance_dict)
	
	exit(0)

if __name__ == '__main__':
	main()







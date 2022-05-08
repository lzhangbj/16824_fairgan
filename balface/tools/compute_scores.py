import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import apricot
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
	'Indian': 'Indian'
}


class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--root', type=str, help='Path to experiment output directory')
	parser.add_argument('--label-file', type=str, default='get all images from label-file')
	parser.add_argument('--real-label-file', type=str, default='save images with its true labels in read_label_file')
	parser.add_argument('--save-name', type=str, help='Path to experiment output directory')
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
			if race == 'Asian': race = 'East Asian'
			race = cls_mapping[race]
			label = cls_list.index(race)
			labels[image_name] = label
	return labels


class DataSet(Dataset):
	def __init__(self, root, label_file, transform):
		super(DataSet, self).__init__()
		self.root = root
		self.transform = transform
		self.image_names = []
		self.labels = []
		with open(label_file, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i==0: continue
				line = line.strip().split(',')
				image_name = line[0]
				cls_id = cls_list.index(cls_mapping[line[1]])
				self.image_names.append(image_name)
				self.labels.append(cls_id)
			
	
	def __len__(self):
		return len(self.image_names)
	
	def __getitem__(self, item):
		image = Image.open(osp.join(self.root, self.image_names[item]))
		image_tensor = self.transform(image)
		return self.image_names[item], image_tensor, self.labels[item]


class ClsDataSet(Dataset):
	def __init__(self, root, label_file, transform, cls_id):
		super(ClsDataSet, self).__init__()
		self.root = root
		self.transform = transform
		self.image_names = []
		self.labels = []
		with open(label_file, 'r') as f:
			for i, line in enumerate(f.readlines()):
				if i == 0: continue
				line = line.strip().split(',')
				image_name = line[0]
				if line[1] == 'Latino_Hispanic': continue
				cls = cls_list.index(cls_mapping[line[1]])
				if cls != cls_id: continue
				self.image_names.append(image_name)
				self.labels.append(cls_id)
	
	def __len__(self):
		return len(self.image_names)
	
	def __getitem__(self, item):
		image = Image.open(osp.join(self.root, self.image_names[item]))
		image_tensor = self.transform(image)
		return self.image_names[item], image_tensor, self.labels[item]


def compute_score(model, data_loader):
	"""
	Compute the score of the indices.

	Parameters
	----------
	model_params: OrderedDict
		Python dictionary object containing models parameters
	idxs: list
		The indices
	"""
	N = 0
	g_is = []
	device='cuda:0'
	embDim = 512
	image_names = []
	labels = []
	print("computing score ... ")
	for batch_idx, (image_name, inputs, targets) in enumerate(tqdm(data_loader)):
		inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
		N += inputs.size()[0]
		l1, out, _ = model.produce_importance(inputs, method='v3')
		loss = F.cross_entropy(out, targets).sum()
		l0_grads = torch.autograd.grad(loss, out)[0]
		g_is.append(l0_grads)
		image_names.extend(image_name)
		labels.append(targets.cpu().numpy())
	
	labels = np.concatenate(labels)
	
	dist_mat = torch.zeros([N, N], dtype=torch.float32)
	first_i = True
	for i, g_i in enumerate(g_is, 0):
		if first_i:
			size_b = g_i.size(0)
			first_i = False
		for j, g_j in enumerate(g_is, 0):
			dist_mat[i * size_b: i * size_b + g_i.size(0),
			j * size_b: j * size_b + g_j.size(0)] = torch.cdist(g_i, g_j).cpu()
	const = torch.max(dist_mat).item()
	dist_mat = (const - dist_mat).numpy()
	return dist_mat, image_names, labels


def compute_gamma(selection_type, dist_mat, idxs):
		"""
		Compute the gamma values for the indices.

		Parameters
		----------
		idxs: list
			The indices

		Returns
		----------
		gamma: list
			Gradient values of the input indices
		"""
		print("computing gammas ... ")
		if selection_type == 'PerClass':
			gamma = [0 for i in range(len(idxs))]
			best = dist_mat[idxs]  # .to(device)
			rep = np.argmax(best, axis=0)
			for i in rep:
				gamma[i] += 1
		elif selection_type == 'Supervised':
			gamma = [0 for i in range(len(idxs))]
			best = dist_mat[idxs]  # .to(device)
			rep = np.argmax(best, axis=0)
			for i in range(rep.shape[1]):
				gamma[rep[0, i]] += 1
		return gamma


def select(budget, args, net, transform):
		"""
		Data selection method using different submodular optimization
		functions.

		Parameters
		----------
		budget: int
			The number of data points to be selected
		model_params: OrderedDict
			Python dictionary object containing models parameters
		optimizer: str
			The optimization approach for data selection. Must be one of
			'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
			'stochastic', 'sample', 'greedi', 'bidirectional'

		Returns
		----------
		total_greedy_list: list
			List containing indices of the best datapoints
		gammas: list
			List containing gradients of datapoints present in greedySet
		"""

		# per_class_bud = int(budget / num_classes)
		total_greedy_list = []
		total_image_names = []
		total_labels = []
		gammas = []

		selection_type = 'PerClass'
		
		if selection_type == 'PerClass':
			for i in range(len(cls_list[1:])):
				i += 1
				dataset = ClsDataSet(args.root, args.label_file, transform, cls_id=i)
				cls_data_loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False, drop_last=False)
				
				dist_mat, image_names, labels = compute_score(net, cls_data_loader)
				fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0,
																				  metric='precomputed',
																				  n_samples=budget,
																				  optimizer='lazy')
				sim_sub = fl.fit_transform(dist_mat)
				greedyList = list(np.argmax(sim_sub, axis=1))
				gamma = compute_gamma(selection_type, dist_mat, greedyList)
				gammas.extend(gamma)
				
				total_image_names.extend([image_names[t] for t in greedyList])
				total_labels.append(labels[np.array(greedyList).astype(np.int32)])
				
				torch.cuda.empty_cache()
			rand_indices = np.random.permutation(len(total_greedy_list))
			total_greedy_list = list(np.array(total_greedy_list)[rand_indices])
			gammas = list(np.array(gammas)[rand_indices])
			
			total_labels = np.concatenate(total_labels)
			
		# elif selection_type == 'Supervised':
		# 	for i in range(num_classes):
		# 		if i == 0:
		# 			idxs = torch.where(labels == i)[0]
		# 			N = len(idxs)
		# 			compute_score(model_params, idxs)
		# 			row = idxs.repeat_interleave(N)
		# 			col = idxs.repeat(N)
		# 			data = dist_mat.flatten()
		# 		else:
		# 			idxs = torch.where(labels == i)[0]
		# 			N = len(idxs)
		# 			compute_score(model_params, idxs)
		# 			row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
		# 			col = torch.cat((col, idxs.repeat(N)), dim=0)
		# 			data = np.concatenate([data, dist_mat.flatten()], axis=0)
		# 	sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(N_trn, N_trn))
		# 	dist_mat = sparse_simmat
		# 	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
		# 																	  n_samples=budget, optimizer=optimizer)
		# 	sim_sub = fl.fit_transform(sparse_simmat)
		# 	total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
		# 	gammas = compute_gamma(total_greedy_list)
		#
		
		
		return gammas, total_image_names, total_labels


def main():
	args = parse_args()
	save_name = args.save_name
	
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

	weights, total_image_names, total_labels = select(5000, args, net, transform)
	
	if args.real_label_file:
		real_labels = read_label_file(args.real_label_file)
		total_labels = []
		for image_name in total_image_names:
			total_labels.append(real_labels[image_name])
		total_labels = np.array(total_labels).astype(np.int32)
	
	with open(f"{save_name}.txt", 'w') as f:
		f.writelines('image_name,race')
		for i in range(len(total_image_names)):
			f.writelines(f"\n{total_image_names[i]},{cls_list[total_labels[i].item()]}")
	
	

if __name__ == '__main__':
	main()



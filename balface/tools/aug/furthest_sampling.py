import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import torch
import torch.nn as nn

cls_list = ['White', 'Black', 'East Asian', 'Indian']

def read_embeddings(file):
	return np.load(file, allow_pickle=True).item()

def read_true_labels(file):
	return np.load(file, allow_pickle=True).item()['true_label']

def get_labels_from_embedding_dict(embeddings_dict):
	label_dict = {}
	for key in embeddings_dict:
		for cls in cls_list:
			if cls in key: break
		label_dict[key] = cls_list.index(cls)
	return label_dict

def get_image_names_list_and_embeddings_np_cls(embeddings_dict, labels_dict):
	ret_dict = {}
	for cls_id, cls in enumerate(cls_list):
		ret_dict[cls] = {
			'embeddings': [],
			'image_names': []
		}
	for image_name, embedding in embeddings_dict.items():
		cls_id = labels_dict[image_name]
		cls = cls_list[cls_id]
		ret_dict[cls]['embeddings'].append(embedding)
		ret_dict[cls]['image_names'].append(image_name)
	for cls_id, cls in enumerate(cls_list):
		ret_dict[cls]['embeddings'] = np.array(ret_dict[cls]['embeddings'])
	return ret_dict

# furthest sampling for each class
def furthest_sampling(distances_np, k):
	selected_indices = [0,]
	distances_np[:, 0] = 0.
	print("furthest sampling")
	for i in tqdm(range(k-1)):
		selected_index = np.argmax(distances_np[selected_indices[-1]])
		distances_np[:, selected_index] = 0.
		selected_indices.append(selected_index)
	return np.array(selected_indices).astype(np.int32)

# furthest sampling for each class
def furthest_sampling_tensor(distances_np, k):
	selected_indices = [0,]
	distances_np[:, 0] = 0.
	print("furthest sampling")
	for i in tqdm(range(k-1)):
		selected_index = torch.argmax(distances_np[selected_indices[-1]])
		distances_np[:, selected_index] = 0.
		selected_indices.append(selected_index)
	return torch.tensor(selected_indices).long().cpu().numpy()

# furthest sampling for each class
def furthest_sampling_topk(distances_np, dist_ids, k):
	selected_indices = np.zeros(1).astype(np.int32)
	B = len(distances_np)
	print("furthest sampling")
	for i in tqdm(range(k-1)):
		prev_id = selected_indices[-1]
		
		curr_distances = torch.zeros(B).float()
		curr_distances.scatter_(dim=0, index=dist_ids[i], src=distances_np[i])
		curr_distances[selected_indices] = 0.
		
		selected_index = torch.argmax(curr_distances)
		
		selected_indices = np.concatenate([selected_indices, selected_index.reshape((1,))]).astype(np.int32)
		
		torch.cuda.empty_cache()
	return np.array(selected_indices).astype(np.int32)

def euclidean_distances_tensor(embeddings1, embeddings2):
	dist = torch.sqrt(torch.sum((embeddings1-embeddings2)**2, dim=1))
	return dist
	

def calc_pairwise_distances_tensor(embeddings_tensor):
	distances = []
	dist_ids = []
	print("calc pairwise distances")
	for i in tqdm(range(0, len(embeddings_tensor), 64)):
		dist = torch.cdist(embeddings_tensor[i:i+64], embeddings_tensor).cpu()
		dist, dist_id = torch.topk(dist, 14000, dim=1)
		distances.append(dist)
		dist_ids.append(dist_id)
		torch.cuda.empty_cache()
	return torch.cat(distances, dim=0), torch.cat(dist_ids, dim=0)

def calc_pairwise_distances(embeddings_np):
	distances = []
	print("calc pairwise distances")
	for i in tqdm(range(0, len(embeddings_np))):
		distances.append(euclidean_distances(embeddings_np[i:i+1], embeddings_np))
	return np.concatenate(distances, axis=0)

with torch.no_grad():
	imb_baseline_embeddings_dict = read_embeddings('./embeddings_scores/fairface_race_imb_14k_embeddings.npy')
	imb_baseline_labels_dict = get_labels_from_embedding_dict(imb_baseline_embeddings_dict)
	
	imb_aug_embeddings_dict = read_embeddings('./embeddings_scores/4race_imb_14k_aug_400k_embeddings.npy')
	imb_aug_labels_dict = get_labels_from_embedding_dict(imb_aug_embeddings_dict)
	imb_aug_true_labels_dict = read_true_labels('./embeddings_scores/4race_imb_14k_aug_400k_acc_labels.npy')
	
	embeddings_dict = imb_baseline_embeddings_dict.copy()
	embeddings_dict.update(imb_aug_embeddings_dict)
	
	labels_dict = imb_baseline_labels_dict.copy()
	labels_dict.update(imb_aug_true_labels_dict)
	
	comb_cls_dict = get_image_names_list_and_embeddings_np_cls(embeddings_dict, labels_dict)
	
	comb_dict = {
		'embeddings': [],
		'image_names': []
	}
	
	
	cls_nums = [0,]
	for cls in cls_list:
		comb_dict['embeddings'].append(comb_cls_dict[cls]['embeddings'])
		comb_dict['image_names'] += comb_cls_dict[cls]['image_names']
		cls_nums.append(cls_nums[-1] + len(comb_cls_dict[cls]['image_names']))
	
	cls_nums = cls_nums[1:]
	
	comb_dict['embeddings'] = torch.from_numpy(np.concatenate(comb_dict['embeddings'], axis=0)).cuda()
	comb_dict['image_names'] = comb_dict['image_names']
	
	selected_dict = {}
	pw_distances, pw_dist_ids = calc_pairwise_distances_tensor(comb_dict['embeddings'])
	np.save("pw_distances", pw_distances.cpu().numpy())
	np.save("pw_dist_ids", pw_dist_ids.cpu().numpy())
	B = len(pw_distances)
	print(pw_distances.size(), pw_dist_ids.size())
	
	# cls_sample_ids = furthest_sampling_topk(pw_distances, pw_dist_ids, 14000)
	cls_sample_ids = furthest_sampling_topk(pw_distances, pw_dist_ids, 14000)
	selected_image_names = []
	
	per_cls_sample_num = [0, 0, 0, 0]
	cls_id_list = []
	for id in cls_sample_ids:
		selected_image_names.append(comb_dict['image_names'][id])
		cls_id = 0
		while cls_id < 4 and id >= cls_nums[cls_id]:
			cls_id += 1
		per_cls_sample_num[cls_id] += 1
		cls_id_list.append(cls_id)
	
	print(f"sampling from cls nums: {per_cls_sample_num}")
	
	with open("furthest_sample_14k.txt", 'w') as f:
		f.writelines("image_names,race")
		for i in range(len(selected_image_names)):
			f.writelines(f"\n{selected_image_names[i]},{cls_list[cls_id_list[i]]}")


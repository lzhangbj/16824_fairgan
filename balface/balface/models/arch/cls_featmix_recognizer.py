import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.models.backbone import build_backbone
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus

class CLSFeatMixRecognizer(nn.Module):
	def __init__(self, model_cfg):
		super(CLSFeatMixRecognizer, self).__init__()
		self.backbone = build_backbone(model_cfg.backbone)
		self.head = build_head(model_cfg.head)
		self.use_weight = model_cfg.use_weight
		self.n_class = model_cfg.head.n_classes[0]
		self.aug_num_per_gpu = model_cfg.aug_num_per_gpu
		self.aug_ratio = np.array(model_cfg.aug_ratio)
		self.aug_ratio /= self.aug_ratio.sum()
		self.aug_method = model_cfg.aug_method

	def forward(self, inputs, labels=None, loss_weight=None, one_hot_labels=False):
		embeddings = self.backbone(inputs)
		
		if not self.training:
			features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight)
			return features_list, loss_dict
	
		if self.aug_method == 'ratio_rand_soft':
			global_embeddings = sync_tensor_across_gpus(embeddings)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]
	
			sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True)
			sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings1 = global_embeddings[sampled1_indices]
			sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class).float()
	
			sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings2 = global_embeddings[sampled2_indices]
			sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class).float()
	
			blend_alphas = np.random.uniform(0.6, 1., size=self.aug_num_per_gpu)
	
			blend_alphas = blend_alphas.astype(np.float32)
			blend_alphas = torch.tensor(blend_alphas).float().cuda().unsqueeze(1)
	
			aug_embeddings = sampled_embeddings2 * (1-blend_alphas) + blend_alphas * sampled_embeddings1
			aug_targets = sampled_targets1 * blend_alphas + sampled_targets2 * (1-blend_alphas)
	
			embeddings = torch.cat([embeddings, aug_embeddings])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

			features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
												 one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]
			
		elif self.aug_method == 'ratio_rand_hard':
			global_embeddings = sync_tensor_across_gpus(embeddings)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets == i).squeeze(1).cpu().numpy() for i in
			                    range(self.n_class)]
			
			sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True)
			sampled_cls1_cls_num_list = [np.sum(sampled_cls1 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in
				 range(self.n_class)])
			sampled_embeddings1 = global_embeddings[sampled1_indices]
			sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class).float()
			
			sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu,
			                                replace=True, p=self.aug_ratio)
			sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in
				 range(self.n_class)])
			sampled_embeddings2 = global_embeddings[sampled2_indices]
			
			blend_alphas = np.random.uniform(0.6, 1., size=self.aug_num_per_gpu)
			
			blend_alphas = blend_alphas.astype(np.float32)
			blend_alphas = torch.tensor(blend_alphas).float().cuda().unsqueeze(1)
			
			aug_embeddings = sampled_embeddings2 * (1 - blend_alphas) + blend_alphas * sampled_embeddings1
			aug_targets = sampled_targets1
			
			embeddings = torch.cat([embeddings, aug_embeddings])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)
			
			features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
			                                     one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]
		
		elif self.aug_method == 'intra_rand':
			global_embeddings = sync_tensor_across_gpus(embeddings)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets == i).squeeze(1).cpu().numpy() for i in
			                    range(self.n_class)]
			
			sampled_cls = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu,
			                                replace=True, p=self.aug_ratio)
			sampled_cls_num_list = [np.sum(sampled_cls == i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls_num_list[i], replace=False) for i in
				 range(self.n_class)])
			sampled_embeddings1 = global_embeddings[sampled1_indices]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls_num_list[i], replace=False) for i in
				 range(self.n_class)])
			sampled_embeddings2 = global_embeddings[sampled2_indices]
			
			sampled_targets = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class).float()
			
			blend_alphas = np.random.uniform(0.6, 1., size=self.aug_num_per_gpu)
			
			blend_alphas = blend_alphas.astype(np.float32)
			blend_alphas = torch.tensor(blend_alphas).float().cuda().unsqueeze(1)
			
			aug_embeddings = sampled_embeddings2 * (1 - blend_alphas) + blend_alphas * sampled_embeddings1
			aug_targets = sampled_targets
			
			embeddings = torch.cat([embeddings, aug_embeddings])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)
			
			features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
			                                     one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]
		else:
			raise Exception()

		return features_list, loss_dict

	def get_embeddings(self, inputs):
		return self.backbone(inputs)
	
	def produce_importance(self, inputs, labels=None, method='v1'):
		B = len(inputs)
		embeddings = self.get_embeddings(inputs)
		logits = self.head.compute_cls_logits(embeddings)[0]
		probs = F.softmax(logits, dim=1)
		
		if method=='v1':
			tgt_prob = probs[torch.arange(B).long(), labels]
			probs[torch.arange(B).long(), labels] = 0.
			max_prob = torch.max(probs, dim=1)[0]
			score = max_prob - tgt_prob
			return embeddings, F.softmax(logits, dim=1), score.view(B)
		if method=='v2':
			sorted_indices = torch.argsort(-probs, dim=1)
			score = torch.gather(probs, dim=1, index=sorted_indices[:, 0:1])
			score_2 = torch.gather(probs, dim=1, index=sorted_indices[:, 1:2])
			score = score_2 - score
			return embeddings, F.softmax(logits, dim=1), score.view(B)
		else:
			raise Exception()




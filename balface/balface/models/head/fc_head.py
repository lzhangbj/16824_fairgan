import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

from balface.utils import GatherLayer, focal_loss, ldam_loss, \
	NormedLinear, sync_tensor_across_gpus, SelfAdaptiveTrainingCE

class ContrastiveHead(nn.Module):
	def __init__(self, embed_dim, input_dim, hidden_dims, T=0.07):
		super(ContrastiveHead, self).__init__()
		assert isinstance(hidden_dims, list)
		self.embed_dim = embed_dim
		self.hidden_dims = hidden_dims

		module_list = []
		in_dim = input_dim
		for i, dim in enumerate(self.hidden_dims):
			module_list.append(nn.Linear(in_dim, dim))
			if i < len(self.hidden_dims)-1:
				module_list.append(nn.ReLU(inplace=True))
			in_dim = dim
		module_list.append(nn.Linear(in_dim, embed_dim))
		self.nonlinear_head = nn.Sequential(*module_list)
		self.T = T

	def contrastive_loss(self, features):
		# input features is normed features
		assert len(features.size()) == 3, features.size()
		B, n_views, dim = features.size()
		assert n_views == 2
		# distributed gathering
		features = torch.cat(GatherLayer.apply(features), dim=0)
		N = features.size(0)
		features = features.view(N, 2, dim).permute(1, 0, 2).contiguous().view(2 * N, dim)  # (2N)xd
		labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)  # (2, N) -> (2*N,)

		rank = dist.get_rank()
		local_indices = np.arange(rank*B, (rank+1)*B).astype(np.int32)
		local_indices = np.concatenate([local_indices, local_indices+N])

		labels = (labels.unsqueeze(0) == labels[local_indices].unsqueeze(1)).float()  # (2B, 2N)
		labels = labels.cuda()

		local_features = features[local_indices]
		similarity_matrix = torch.matmul(local_features, features.T)  # (2B,2N)

		# discard the main diagonal from both: labels and similarities matrix
		mask = torch.eye(2*N, dtype=torch.bool)[local_indices].cuda()

		labels = labels[~mask].view(2*B, -1)  # (2B, 2N-1)
		similarity_matrix = similarity_matrix[~mask].view(2*B, -1)  # (2B, 2N-1)

		# select and combine multiple positives
		positives = similarity_matrix[labels.bool()].view(2*B, -1)
		# select only the negatives the negatives
		negatives = similarity_matrix[~labels.bool()].view(2*B, -1)

		logits = torch.cat([positives, negatives], dim=1)
		labels = torch.zeros(2*B).long().cuda()

		logits = logits / self.T

		contrast_loss = F.cross_entropy(logits, labels)

		return contrast_loss

	def forward(self, input1, input2):
		B = input1.size(0)
		embeddings1 = self.nonlinear_head(input1)
		embeddings2 = self.nonlinear_head(input2)

		features = torch.stack([embeddings1, embeddings2], dim=1) # (B, 2, dim)
		features = F.normalize(features, dim=2)

		features1, features2 = features[:B], features[B:]

		loss_dict = {}
		if self.training:
			contrast_loss = self.contrastive_loss(features)
			loss_dict = {"ssp_loss": contrast_loss}

		return [features1, features2], loss_dict

	def compute_ssp_embedding(self, input):
		assert not self.training
		with torch.no_grad():
			embedding = F.normalize(
				self.nonlinear_head(input), dim=1
			)
		return embedding


class ClassifierHead(nn.Module):
	'''
		discarded
		always use MultiClassifierHead with len(n_classes) = 1
	'''
	def __init__(self, n_class, input_dim, hidden_dims, loss='ce', norm_weights=False):
		super(ClassifierHead, self).__init__()
		assert isinstance(hidden_dims, list)
		self.n_class = n_class
		self.hidden_dims = hidden_dims
		self.loss = loss
		if self.loss in ['ce', 'ib-ce']:
			self.loss_func = F.cross_entropy
		elif self.loss == 'kld':
			self.loss_func = F.kl_div
		elif self.loss == 'focal':
			self.loss_func = focal_loss
		elif self.loss == 'ldam':
			self.loss_func = ldam_loss
		else:
			raise Exception()
		# self.loss_func = FocalLoss(size_average=False)

		module_list = []
		in_dim = input_dim
		for i, dim in enumerate(self.hidden_dims):
			module_list.append(nn.Linear(in_dim, dim))
			module_list.append(nn.Dropout(0.5))
			module_list.append(nn.ReLU(inplace=True))
			in_dim = dim
		if norm_weights:
			module_list.append(NormedLinear(in_dim, n_class))
		else:
			module_list.append(nn.Linear(in_dim, n_class))
		self.nonlinear_head = nn.Sequential(*module_list)
		self.norm_weights = norm_weights

		self.margins = torch.tensor([0.1,0.2,0.2,0.2]).float()
	
	def get_per_sample_weights(self, logits):
		prob_logits = F.softmax(logits, dim=1)
		# prob_logits[0][:, 0] *= 1.0
		# prob_logits[0] = prob_logits[0] / torch.sum(prob_logits[0], dim=1, keepdim=True)
		# pseudo_labels = torch.stack([torch.argmax(prob_logit, dim=1) for prob_logit in prob_logits], dim=1)
		# prob_probs = prob_logits[0][torch.arange(len(prob_logits[0])).long().cuda(), pseudo_labels[:, 0]]
		# weights = (1-prob_probs) ** 2
		# # weights[pseudo_labels[:, 0]==0] = 0.
		#
		# weights = weights / weights.sum()
		# mask = weights.unsqueeze(1)
		
		# probs = [torch.max(prob_logit, dim=1)[0] for prob_logit in prob_logits]
		probs_sorted = torch.sort(prob_logits, dim=1, descending=True)[0]
		uncertainties = probs_sorted[:, 1] - probs_sorted[:, 0]
		uncertainties_max = torch.max(uncertainties, dim=0, keepdim=True)[0]
		uncertainties_min = torch.min(uncertainties, dim=0, keepdim=True)[0]
		uncertainties_max = torch.max(sync_tensor_across_gpus(uncertainties_max), dim=0, keepdim=True)[0]
		uncertainties_min = torch.min(sync_tensor_across_gpus(uncertainties_min), dim=0, keepdim=True)[0]
		
		uncertainties = (uncertainties - uncertainties_min) / (uncertainties_max - uncertainties_min)
		uncertainties = uncertainties ** 0.5
		
		mask = uncertainties / uncertainties.sum()
		
		return mask

	def forward(self, labeled_input, unlabeled_input=None, labels=None, pseudo_labels=None, mask=None, label_weights=None, unlabel_weights=None, one_hot_labels=False):
		loss_dict = {}

		logits_list = []

		labeled_logits = self.nonlinear_head(labeled_input)
		
		# weights = self.get_per_sample_weights(labeled_logits)
		
		logits_list.append(labeled_logits)
		if self.training:
			# margins = self.margins[labels].cuda()
			# labeled_logits[torch.arange(len(labels)).long().cuda(), labels] -= margins
			# labeled_logits = labeled_logits.contiguous()
			
			loss = self.loss_func(labeled_logits, labels.contiguous(), weight=label_weights, reduction='none')
			if self.loss == 'ib-ce':
				h_norm = torch.sum(torch.abs(labeled_input), dim=1)
				probs = F.softmax(labeled_logits, dim=1)
				target_one_hot = F.one_hot(labels, num_classes=self.n_class).float().cuda()
				prob_diff_norm = torch.sum(torch.abs(probs-target_one_hot), dim=1)
				sample_weights = 1./(prob_diff_norm * h_norm)

				## class-wise
				# calculate global class weights sum
				targets_one_hot = F.one_hot(labels, num_classes=self.n_class).float().cuda()
				class_wise_sample_weights = sample_weights.unsqueeze(1) * targets_one_hot
				class_wise_sample_weights_sum = class_wise_sample_weights.sum(dim=0, keepdim=True)
				class_wise_sample_weights_sum = torch.sum(sync_tensor_across_gpus(class_wise_sample_weights_sum), dim=0) # (n_class, )
				# calculate global class gt sum
				class_targets_sum = torch.sum(targets_one_hot, dim=0, keepdim=True)
				class_targets_sum = torch.sum(sync_tensor_across_gpus(class_targets_sum), dim=0) # (n_class, )

				samplewise_num_class_samples = class_targets_sum[labels] # (B, )
				samplewise_weights_sum = class_wise_sample_weights_sum[labels]

				sample_weights = samplewise_num_class_samples * sample_weights / (samplewise_weights_sum + 1e-7)

				# ## sample-wise
				# B = labeled_input.size(0) * dist.get_world_size()
				# sample_wise_sample_weights_sum = torch.sum(sync_tensor_across_gpus(sample_weights))
				# sample_weights = B * sample_weights / (sample_wise_sample_weights_sum + 1e-7)

				# print(sample_weights.mean().detach().cpu().numpy(), sample_weights.max().detach().cpu().numpy(), sample_weights.min().detach().cpu().numpy())
				loss *= sample_weights

			loss_dict["cls_loss"] = loss

		if unlabeled_input is not None:
			unlabeled_logits = self.nonlinear_head(unlabeled_input)
			
			# margins = self.margins[pseudo_labels].cuda()
			# unlabeled_logits[torch.arange(len(pseudo_labels)).long().cuda(), pseudo_labels] -= margins
			# unlabeled_logits = unlabeled_logits.contiguous()
			
			logits_list.append(unlabeled_logits)
			if self.training:
				if mask is not None:
					loss = self.loss_func(unlabeled_logits, pseudo_labels.contiguous(),
													reduction='none', weight=unlabel_weights) * mask / (1e-6 + torch.sum(mask))
				else:
					loss = self.loss_func(unlabeled_logits, pseudo_labels.contiguous(), weight=unlabel_weights, reduction='none')

				loss_dict["ssl_loss"] = loss

		return logits_list, loss_dict

	def compute_cls_logits(self, input, no_grad=True):
		assert not self.training
		if no_grad:
			with torch.no_grad():
				logits = self.nonlinear_head(input)
		else:
			logits = self.nonlinear_head(input)
		return logits

	@torch.no_grad()
	def norm_classifier_weights(self):
		for mod in self.nonlinear_head:
			if isinstance(mod, nn.Linear):
				w = mod.weight.data
				w = F.normalize(w, dim=1)
				mod.weight.data = w


class MultiClassifierHead(nn.Module):
	def __init__(self, n_classes, input_dim, hidden_dims, loss='ce', norm_weights=False):
		super(MultiClassifierHead, self).__init__()
		self.n_classifier = len(n_classes)
		self.loss = loss

		for i, n_class in enumerate(n_classes):
			setattr(self, f'classifier_{i+1}', ClassifierHead(n_class, input_dim, hidden_dims, loss=loss, norm_weights=norm_weights))

	def forward(self, labeled_input, unlabeled_input=None, labels=None, pseudo_labels=None, masks=None, label_weights=None, unlabel_weights=None, one_hot_labels=False):
		logits_list = []
		loss_dict = {}
		for i in range(self.n_classifier):
			getattr(self, f'classifier_{i + 1}').loss = self.loss

			per_logit_list, per_loss_dict = getattr(self, f'classifier_{i+1}')(labeled_input, unlabeled_input,
																			   labels[:, i] if labels is not None else None,
																			   pseudo_labels[:, i] if pseudo_labels is not None else None,
																			   masks[:, i] if masks is not None else None,
																			   label_weights=label_weights if i==0 else None,
																			   unlabel_weights=unlabel_weights if i==0 else None,
																			   one_hot_labels=one_hot_labels)
			logits_list.extend(per_logit_list)
			if self.training:
				loss_dict[f"cls_{i + 1}_loss"] = per_loss_dict['cls_loss']
				if unlabeled_input is not None:
					loss_dict[f"ssl_{i + 1}_loss"] = per_loss_dict['ssl_loss']
		return logits_list, loss_dict

	def compute_cls_logits(self, input, no_grad=True):
		assert not self.training
		logits = []
		for i in range(self.n_classifier):
			logits.append(getattr(self, f'classifier_{i+1}').compute_cls_logits(input, no_grad=no_grad))
		return logits

	@torch.no_grad()
	def norm_classifier_weights(self):
		for i in range(self.n_classifier):
			getattr(self, f'classifier_{i+1}').norm_classifier_weights()


class MultiClassifierContrastiveHead(nn.Module):
	def __init__(self, n_classes, input_dim, cls_hidden_dims, embed_dim, contrast_hidden_dims, T=0.07, loss='ce'):
		super(MultiClassifierContrastiveHead, self).__init__()
		self.n_classifier = len(n_classes)
		self.embed_dim = embed_dim
		self.T = T
		self.loss = loss

		self.multiclassifier_head = MultiClassifierHead(n_classes, input_dim, cls_hidden_dims, loss=loss)
		self.contrastive_head = ContrastiveHead(embed_dim, input_dim, contrast_hidden_dims, T)

	def forward(self, labeled_input1, labeled_input2=None,
				unlabeled_input1=None, unlabeled_input2=None,
				labels=None, pseudo_labels=None, masks=None,
				label_weights=None, unlabel_weights=None):
		# ssl
		ssl_logits_list, ssl_loss_dict = self.multiclassifier_head(labeled_input1, unlabeled_input1,
																   labels, pseudo_labels, masks,
																   label_weights, unlabel_weights)
		# ssp
		view1 = torch.cat([labeled_input1, unlabeled_input1])
		view2 = torch.cat([labeled_input2, unlabeled_input2])
		ssp_embeddings_list, ssp_loss_dict = self.contrastive_head(view1, view2)

		feat_list = ssl_logits_list + ssp_embeddings_list
		ssl_loss_dict.update(ssp_loss_dict)

		return feat_list, ssl_loss_dict

	def compute_cls_logits(self, input):
		return self.multiclassifier_head.compute_cls_logits(input)

	def compute_ssp_embeddings(self, input):
		return self.contrastive_head.compute_ssp_embedding(input)



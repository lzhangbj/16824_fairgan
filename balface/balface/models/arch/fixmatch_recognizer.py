import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from balface.models.backbone import build_backbone
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus


class FixMatchRecognizer(nn.Module):
	def __init__(self, model_cfg):
		super(FixMatchRecognizer, self).__init__()
		self.backbone = build_backbone(model_cfg.backbone)
		self.head = build_head(model_cfg.head)
		n_class = model_cfg.head.n_classes[0]
		self.n_class = n_class
		self.use_covariance_matrix = False
		self.use_conditional_matrix = False
		if 'use_covariance_matrix' in model_cfg and model_cfg.use_covariance_matrix:
			self.use_covariance_matrix = True
			covariance_matrix = torch.ones((n_class, n_class)).cuda()
			self.covariance_matrix = nn.Parameter(covariance_matrix, requires_grad=False)

		if "use_conditional_matrix" in model_cfg and model_cfg.use_conditional_matrix:
			self.use_conditional_matrix = True
			conditional_matrix = torch.eye(n_class)
			conditional_std = torch.zeros((n_class, n_class))
			self.conditional_matrix = nn.Parameter(conditional_matrix, requires_grad=False)
			self.conditional_std = nn.Parameter(conditional_std, requires_grad=False)

			self.momentum_alpha = model_cfg.momentum_alpha

		self.ema_backbone = build_backbone(model_cfg.backbone)
		self.ema_head = build_head(model_cfg.head)
		self.ema_momentum = model_cfg.ema_momentum
		for param in self.ema_backbone.parameters():
			param.detach_()
			param.requires_grad = False
		for param in self.ema_head.parameters():
			param.detach_()
			param.requires_grad = False

		self.loss = model_cfg.loss
		self.use_weight = model_cfg.use_weight
		if self.use_weight:
			weight = torch.ones(7).float().cuda()/7.
			self.weights = nn.Parameter(weight, requires_grad=False)
			self.weight_momentum = 0.9

	@torch.no_grad()
	def update_ema_module(self, model, shadow):
		if not self.training:
			print("EMA update should only be called during training", file=stderr, flush=True)
			return

		model_params = OrderedDict(model.named_parameters())
		shadow_params = OrderedDict(shadow.named_parameters())

		# check if both model contains the same set of keys
		assert model_params.keys() == shadow_params.keys()

		for name, param in model_params.items():
			# see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
			# shadow_variable -= (1 - decay) * (shadow_variable - variable)
			shadow_params[name].sub_((1. - self.ema_momentum) * (shadow_params[name] - param))

		model_buffers = OrderedDict(model.named_buffers())
		shadow_buffers = OrderedDict(shadow.named_buffers())

		# check if both model contains the same set of keys
		assert model_buffers.keys() == shadow_buffers.keys()

		for name, buffer in model_buffers.items():
			# buffers are copied
			shadow_buffers[name].copy_(buffer)

	def update_ema(self):
		self.update_ema_module(self.backbone, self.ema_backbone)
		self.update_ema_module(self.head, self.ema_head)

	def get_embeddings(self, inputs):
		return self.ema_backbone(inputs)

	def init_ema(self):
		self.ema_backbone.load_state_dict(self.backbone.state_dict())
		self.ema_head.load_state_dict(self.head.state_dict())
		for param in self.ema_backbone.parameters():
			param.requires_grad = False
		for param in self.ema_head.parameters():
			param.requires_grad = False

	def update_conditional_matrix(self, probs, targets):
		probs = probs.contiguous()
		targets = targets.contiguous()
		probs = sync_tensor_across_gpus(probs).cpu()
		targets = sync_tensor_across_gpus(targets).cpu()

		preds = probs.argmax(dim=1)
		n_class = probs.size(1)
		for i in range(n_class):
			for j in range(n_class):
				preds_i_mask = preds == i
				targets_j_mask = targets == j
				preds_i_targets_j_mask = preds_i_mask & targets_j_mask
				conditional_ij_prob = probs[preds_i_targets_j_mask].mean()
				conditional_ij_std = probs[preds_i_targets_j_mask].std()

				self.conditional_matrix[i, j] = \
					self.momentum_alpha * self.conditional_matrix[i, j] + (1-self.momentum_alpha) * conditional_ij_prob.cuda()

				self.conditional_std[i, j] = \
					self.momentum_alpha * self.conditional_std[i, j] + (1 - self.momentum_alpha) * conditional_ij_std.cuda()

	def update_covariance_matrix(self, probs, targets):
		probs = probs.contiguous()
		targets = targets.contiguous()
		probs = sync_tensor_across_gpus(probs).cpu()
		targets = sync_tensor_across_gpus(targets).cpu()

		preds = probs.argmax(dim=1)
		n_class = probs.size(1)
		for i in range(n_class):
			for j in range(n_class):
				preds_i_mask = preds == i
				targets_j_mask = targets == j
				preds_i_targets_j_mask = preds_i_mask & targets_j_mask
				covariance_ij_prob = (preds_i_targets_j_mask.double().sum() / preds_i_mask.double().sum())

				self.covariance_matrix[i, j] = \
					self.momentum_alpha * self.covariance_matrix[i, j] + (
								1 - self.momentum_alpha) * covariance_ij_prob.cuda()

	def correct_probs_with_covariance_matrix(self, probs):
		max_cls = probs.argmax(dim=1)
		covariance_vectors = self.covariance_matrix[max_cls]
		corrected_probs = F.normalize(probs * covariance_vectors, dim=1)
		return corrected_probs

	def compute_pseudolabel(self, logits, thresh=0.95):
		if self.use_conditional_matrix:
			assert not self.use_covariance_matrix
			assert self.loss == 'kld'
			pseudo_labels = torch.stack([torch.argmax(logit, dim=1) for logit in logits], dim=1)
			probs = [torch.max(torch.softmax(logit, dim=1), dim=1)[0] for logit in logits]
			mask = torch.stack([prob >= thresh for prob in probs], dim=1).double()

			logits = logits[0]
			pseudo_labels = pseudo_labels[:, 0]
			mask = mask[:, 0]

			probs = F.softmax(logits, dim=1).detach()
			probs[:, pseudo_labels] = 0.
			top1_probs, top1_cls = torch.max(probs, dim=1)
			top1_criterion = self.conditional_matrix[pseudo_labels, top1_cls] - self.conditional_std[pseudo_labels, top1_cls]
			pseudo_labels = torch.where(top1_probs >= top1_criterion, top1_cls, pseudo_labels)
			mask = torch.where(top1_probs >= top1_criterion, mask, 1.0)

			pseudo_labels = pseudo_labels.unsqueeze(1)
			mask = mask.unsqueeze(1)
		elif self.use_covariance_matrix:
			if self.loss == 'ce':
				pseudo_labels = torch.stack([torch.argmax(logit, dim=1) for logit in logits], dim=1)
				probs = [torch.max(torch.softmax(logit, dim=1), dim=1)[0] for logit in logits]
				mask = torch.stack([prob >= thresh for prob in probs], dim=1).float()
			elif self.loss == 'kld':
				pseudo_labels = torch.softmax(logits[0], dim=1) # (B, 7)
				max_cls = logits[0].argmax(dim=1)

				covariance_weights = self.covariance_matrix[max_cls]
				# covariance_weights = (cls_one_hot.unsqueeze(2) * self.covariance_matrix.unsqueeze(0)).sum(dim=1)

				pseudo_labels = F.normalize(pseudo_labels * covariance_weights, dim=1)
				max_probs = torch.max(pseudo_labels, dim=1)[0]
				mask = (max_probs >= 0).float() # remove mask

				pseudo_labels = pseudo_labels.unsqueeze(1)

				mask = mask.unsqueeze(1)
		else:
			if self.loss == 'ce':
				prob_logits = [F.softmax(logit, dim=1) for logit in logits]
				# prob_logits[0][:, 0] *= 1.0
				# prob_logits[0] = prob_logits[0] / torch.sum(prob_logits[0], dim=1, keepdim=True)
				pseudo_labels = torch.stack([torch.argmax(prob_logit, dim=1) for prob_logit in prob_logits], dim=1)
				# prob_probs = prob_logits[0][torch.arange(len(prob_logits[0])).long().cuda(), pseudo_labels[:, 0]]
				# weights = (1-prob_probs) ** 2
				# # weights[pseudo_labels[:, 0]==0] = 0.
				#
				# weights = weights / weights.sum()
				# mask = weights.unsqueeze(1)
				
				probs = [torch.max(prob_logit, dim=1)[0] for prob_logit in prob_logits]
				mask = torch.stack([prob >= 0 for prob in probs], dim=1).float()
				# mask_one = (pseudo_labels != 0).float()
				# probs_sorted = [torch.sort(prob_logit, dim=1, descending=True)[0] for prob_logit in prob_logits]
				# uncertainties = [prob_sorted[:, 1] - prob_sorted[:, 0] for prob_sorted in probs_sorted]
				# uncertainties_max = torch.stack([torch.max(uncertainty, dim=0, keepdim=True)[0] for uncertainty in uncertainties], dim=1)
				# uncertainties_min = torch.stack([torch.min(uncertainty, dim=0, keepdim=True)[0] for uncertainty in uncertainties], dim=1)
				# uncertainties_max = torch.max(sync_tensor_across_gpus(uncertainties_max), dim=0, keepdim=True)[0]
				# uncertainties_min = torch.min(sync_tensor_across_gpus(uncertainties_min), dim=0, keepdim=True)[0]
				#
				# uncertainties = torch.stack(uncertainties, dim=1)
				# uncertainties = (uncertainties - uncertainties_min) / (uncertainties_max - uncertainties_min)
				# # uncertainties = uncertainties ** 0.5
				#
				# mask = uncertainties / uncertainties.sum()
				# mask = torch.minimum(mask_one, uncertainties)
				
			elif self.loss == 'kld':
				pseudo_labels = torch.softmax(logits[0], dim=1)  # (B, 7)
				max_probs = torch.max(pseudo_labels, dim=1)[0]
				mask = (max_probs >= thresh).float()

				pseudo_labels = pseudo_labels.unsqueeze(1)
				mask = mask.unsqueeze(1)

		return pseudo_labels, mask

	def forward(self, label_input, weak_input=None, strong_input=None, labels=None, threshold=0.95, label_weights=None, unlabel_weights=None):
		if self.training:
			if weak_input is None:
				embeddings = self.backbone(label_input)
				features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=None)
				return features_list, loss_dict

			B_labeled = label_input.size(0)
			B_unlabeled = weak_input.size(0)
			input = torch.cat([label_input, strong_input])
			embeddings = self.backbone(input)

			labeled_embeddings = embeddings[:B_labeled]
			strong_embeddings = embeddings[-B_unlabeled:]

			with torch.no_grad():
				self.eval()
				weak_embeddings = self.ema_backbone(weak_input)
				logits = self.ema_head.compute_cls_logits(weak_embeddings)
			pseudo_labels, masks = self.compute_pseudolabel(logits, threshold)

			# label_weights = None
			# if self.use_weight:
			#     if unlabel_weights is None:
			#         dist_pseudo_labels = sync_tensor_across_gpus(pseudo_labels)
			#         squeezed_pseudo_labels = dist_pseudo_labels.squeeze(1)
			#         one_hot_pseudo_labels = torch.eye(7).float().cuda()[squeezed_pseudo_labels] # (B, 7)
			#         updated_weights = 1. / (one_hot_pseudo_labels.sum(dim=0) / one_hot_pseudo_labels.sum() + 1e-6)
			#         updated_weights = updated_weights / updated_weights.sum()
			#         self.weights.data = self.weight_momentum*self.weights.data + updated_weights*(1-self.weight_momentum)
			#         self.weights.data = self.weights.data / self.weights.data.sum()
			#         unlabel_weights = self.weights.data
			#
			#     label_weights = torch.tensor([1., 10, 10, 10, 10, 10, 10.]).float().cuda()
			#     label_weights = label_weights / label_weights.sum()
			#     unlabel_weights = label_weights
			self.train()

			# compute pseudo_labels adaptively if it is not provided
			if self.loss == 'ce':
				features_list, loss_dict = self.head(labeled_embeddings, strong_embeddings, labels, pseudo_labels, masks,
													 label_weights=label_weights, unlabel_weights=unlabel_weights)
			elif self.loss == 'kld':
				features_list, loss_dict = self.head(labeled_embeddings, strong_embeddings, labels, pseudo_labels,
													 masks, label_weights=label_weights, unlabel_weights=unlabel_weights)
			else:
				raise Exception()

			if self.use_conditional_matrix:
				self.update_conditional_matrix(F.softmax(features_list[0], dim=1), labels[:, 0])

			return features_list, loss_dict, pseudo_labels
		else:
			embeddings = self.ema_backbone(label_input)
			logits = self.ema_head.compute_cls_logits(embeddings)
			probs = [F.softmax(logit, dim=1) for logit in logits]
			if self.use_covariance_matrix:
				probs = [self.correct_probs_with_covariance_matrix(prob) for prob in probs]
			return probs, {}

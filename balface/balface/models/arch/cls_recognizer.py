import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.utils import sync_tensor_across_gpus
from balface.models.backbone import build_backbone
from balface.models.head import build_head

class CLSRecognizer(nn.Module):
	def __init__(self, model_cfg):
		super(CLSRecognizer, self).__init__()
		self.backbone = build_backbone(model_cfg.backbone)
		self.head = build_head(model_cfg.head)
		self.use_weight = model_cfg.use_weight

	def forward(self, inputs, labels=None, loss_weight=None, one_hot_labels=False):
		embeddings = self.backbone(inputs)

		label_weights = None

		features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight, one_hot_labels=one_hot_labels)

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
		if method=='v3':
			with torch.no_grad():
				embeddings = self.get_embeddings(inputs)
			logits = self.head.compute_cls_logits(embeddings, no_grad=False)[0]
			probs = F.softmax(logits, dim=1)
			
			sorted_indices = torch.argsort(-probs, dim=1)
			score = torch.gather(probs, dim=1, index=sorted_indices[:, 0:1])
			score_2 = torch.gather(probs, dim=1, index=sorted_indices[:, 1:2])
			score = score_2 - score
			return embeddings, logits, score.view(B)
		else:
			raise Exception()




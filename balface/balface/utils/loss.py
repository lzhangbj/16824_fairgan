import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, target, weight=None, reduction='none'):
	pt = F.softmax(logits, dim=1)
	pt = torch.index_select(pt, 1, target).squeeze(1)

	loss = F.cross_entropy(logits, target, weight=weight, reduction='none')
	loss = loss * (1-pt)**2

	if reduction == 'none':
		return loss
	return loss.mean()

def ldam_loss(logits, target, weight=None, reduction='none'):
	ldam_margin = torch.tensor([0.1, 1, 1, 1, 1, 1, 1]).cuda()**0.25
	max_margin = ldam_margin.max()
	ratio = 0.5 / max_margin

	ldam_margin *= ratio

	index = torch.zeros_like(logits, dtype=torch.uint8)
	index.scatter_(1, target.unsqueeze(1), 1)

	index_float = index.float().cuda()
	batch_m = torch.matmul(ldam_margin.unsqueeze(0), index_float.transpose(0, 1))
	batch_m = batch_m.view((-1, 1))
	x_m = logits - batch_m

	output = torch.where(index, x_m, logits)

	return F.cross_entropy(output, target, weight=weight, reduction=reduction)


class SelfAdaptiveTrainingCE():
	def __init__(self, labels, num_classes=4, momentum=0.9):
		# initialize soft labels to onthot vectors
		self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
		self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
		self.momentum = momentum
	
	def update_labels(self, logits, index):
		prob = F.softmax(logits.detach(), dim=1)
		self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob
	
	def __call__(self, logits, index):
		# obtain prob, then update running avg
		# obtain weights
		weights, _ = self.soft_labels[index].max(dim=1)
		weights *= logits.shape[0] / weights.sum()
		
		# compute cross entropy loss, without reduction
		loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)
		
		# sample weighted mean
		# loss = loss * weights
		return loss
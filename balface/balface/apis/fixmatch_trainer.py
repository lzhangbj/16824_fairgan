# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings
from collections import OrderedDict
import time
from datetime import timedelta
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import shutil

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from balface.models import build_model
from balface.datasets import build_dataset
from balface.utils import get_root_logger, AverageMeter, accuracy, SSLDataloader, sync_tensor_across_gpus

from warmup_scheduler import GradualWarmupScheduler


def build_optimizer(model, opt_cfg):
	assert opt_cfg.name in ['sgd', 'adam', 'adamw']
	lr = opt_cfg.lr
	weight_decay = opt_cfg.weight_decay
	if opt_cfg.name == 'sgd':
		momentum = opt_cfg.momentum
		parameters = [param for param in  model.parameters() if param.requires_grad]
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif opt_cfg.name == 'adam':
		parameters = [param for param in model.parameters() if param.requires_grad]
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif opt_cfg.name == 'adamw':
		parameters = [param for param in model.parameters() if param.requires_grad]
		optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	return optimizer


def build_lr_scheduler(scheduler_cfg, optimizer, last_epoch):
	assert scheduler_cfg.name in ['multisteplrwwarmup', 'multisteplr']
	if scheduler_cfg.name == 'multisteplrwwarmup':
		milestones = scheduler_cfg.milestones
		after_warmup = MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=last_epoch)
		scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=scheduler_cfg.warmup_epoch, after_scheduler=after_warmup)
		return scheduler_warmup
	elif scheduler_cfg.name == 'multisteplr':
		milestones = scheduler_cfg.milestones
		return MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=last_epoch)


class FixMatchTrainer:
	def __init__(self, cfg):
		self.cfg = cfg

		self.logger = cfg.logger
		if dist.get_rank() == 0:
			self.writer = SummaryWriter(f"{self.cfg.work_dir}/tfb")
		self.epoch = -1

		model = build_model(cfg.model)
		find_unused_parameters = cfg.get('find_unused_parameters', False)
		self.model = DistributedDataParallel(
			model.cuda(),
			device_ids=[torch.cuda.current_device()],
			find_unused_parameters=find_unused_parameters)

		self.optimizer = build_optimizer(model, cfg.optimizer)
		if cfg.resume_from:
			self.logger.info(f"resume from {cfg.resume_from}")
			self.resume(cfg.resume_from)
		if cfg.load_backbone:
			self.logger.info(f"loading backbone from {cfg.load_backbone}")
			self.load_backbone(cfg.load_backbone)
		if cfg.load_from:
			self.logger.info(f"loading model from {cfg.load_from}")
			self.load_from(cfg.load_from)

		self.model.module.update_ema()

		self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, self.optimizer, last_epoch=self.epoch)

		labeled_train_dataset = build_dataset(cfg.data.labeled_train)
		self.logger.info(f"########## training with {len(labeled_train_dataset)} labeled data ###########")
		labeled_sampler = DistributedSampler(labeled_train_dataset, shuffle=True)
		self.labeled_dataloader = DataLoader(labeled_train_dataset,
											 sampler=labeled_sampler,
											 batch_size=cfg.data.labeled_train.samples_per_gpu,
											 num_workers=cfg.data.labeled_train.workers_per_gpu,
											 drop_last=False)

		unlabeled_train_dataset = build_dataset(cfg.data.unlabeled_train)
		self.logger.info(f"########## training with {len(unlabeled_train_dataset)} unlabeled data ###########")
		unlabeled_sampler = DistributedSampler(unlabeled_train_dataset, shuffle=True)
		self.unlabeled_dataloader = DataLoader(unlabeled_train_dataset,
											   sampler=unlabeled_sampler,
											   batch_size=cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		val_dataset = build_dataset(cfg.data.val)
		# prepare data loaders
		self.logger.info(f"########## evaluating with {len(val_dataset)} val_dataset data ###########")
		self.val_loader = DataLoader(
			val_dataset,
			batch_size=cfg.data.val.samples_per_gpu,
			num_workers=cfg.data.val.workers_per_gpu,
			drop_last=False
		)
		
		try:
			test_dataset = build_dataset(cfg.data.test)
			# prepare data loaders
			self.logger.info(f"########## testing with {len(test_dataset)} val_dataset data ###########")
			self.test_loader = DataLoader(
				test_dataset,
				batch_size=cfg.data.test.samples_per_gpu,
				num_workers=cfg.data.test.workers_per_gpu,
				drop_last=False
			)
		except:
			test_dataset = build_dataset(cfg.data.val)
			# prepare data loaders
			self.logger.info(f"########## testing with {len(test_dataset)} val_dataset data ###########")
			self.test_loader = DataLoader(
				test_dataset,
				batch_size=cfg.data.val.samples_per_gpu,
				num_workers=cfg.data.val.workers_per_gpu,
				drop_last=False
			)

		# n_class = self.cfg.model.head.n_classes[0]
		# self.cls_loss_weight = torch.tensor([0., 0.07, 0.18, 0.25, 0.11, 0.18, 0.20], requires_grad=False).cuda()
		# self.cls_loss_weight = (self.cls_loss_weight + 0.0001)
		# self.cls_loss_weight /= self.cls_loss_weight.sum()
		n_class = self.cfg.model.head.n_classes[0]
		self.cls_loss_weight = torch.ones(n_class, requires_grad=False).cuda()
		self.cls_loss_weight[0] = 0.4
		# self.cls_loss_weight[3] = 3.0
		self.cls_loss_weight = self.cls_loss_weight.contiguous()
		# self.cls_loss_weight = self.cls_loss_weight / self.cls_loss_weight.sum()
		
		self.unlabeled_cls_loss_weight = torch.ones(n_class, requires_grad=False).cuda()
		self.unlabeled_cls_loss_weight[0] = 0.4
		# self.cls_loss_weight[3] = 3.0
		self.unlabeled_cls_loss_weight = self.unlabeled_cls_loss_weight.contiguous()
		# self.unlabeled_cls_loss_weight = self.unlabeled_cls_loss_weight / self.unlabeled_cls_loss_weight.sum()

	def resume(self, resume_from):
		save_content = torch.load(resume_from, map_location='cpu')
		epoch = save_content['epoch']
		statedict = save_content['state_dict']
		opt_state = save_content['opt_state']

		self.model.module.load_state_dict(statedict)
		self.optimizer.load_state_dict(opt_state)
		self.epoch = epoch

	def load_backbone(self, load_from):
		save_content = torch.load(load_from, map_location='cpu')
		statedict = save_content['state_dict']
		backbone_state_dict = OrderedDict()
		for key in statedict.keys():
			if 'backbone' in key:
				backbone_state_dict[key] = statedict[key]
		self.model.module.load_state_dict(backbone_state_dict, strict=False)

	def load_from(self, load_from):
		save_content = torch.load(load_from, map_location='cpu')
		statedict = save_content['state_dict']
		self.model.module.load_state_dict(statedict)

	def save_checkpoint(self, latest=True, best=False):
		save_dict = {
			'epoch': self.epoch,
			'state_dict': self.model.module.state_dict(),
			'opt_state': self.optimizer.state_dict()
		}
		save_file = f"{self.cfg.work_dir}/epoch_{self.epoch}.pth"
		torch.save(save_dict, save_file)
		if latest:
			latest_file = f"{self.cfg.work_dir}/latest.pth"
			shutil.copyfile(save_file, latest_file)
		if best:
			best_file = f"{self.cfg.work_dir}/best_model.pth"
			shutil.copyfile(save_file, best_file)

	def resample_unlabeled_fixed_ratio(self):
		dataset = self.unlabeled_dataloader.dataset
		n_classifier = len(self.cfg.data.labeled_train.n_classes)
		ratios = [0.1**0.333, 1., 1., 1., 1., 1., 1.]
		# create pseudo label
		dataset.gen_pseudolabel = True

		all_image_indices = []
		all_pseudo_labels = []
		all_probs = []

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)


		for i, unlabeled_batch in enumerate(self.unlabeled_dataloader):
			weak_inputs, image_index = unlabeled_batch
			inputs = weak_inputs.cuda()
			image_index = image_index.cuda()
			probs, _ = self.model(inputs)

			probs = probs[0]
			pseudo_labels = probs.argmax(dim=1)
			probs = torch.max(probs.softmax(dim=1), dim=1)[0]

			torch.cuda.empty_cache()

			all_image_indices.append(image_index)
			all_pseudo_labels.append(pseudo_labels)
			all_probs.append(probs)

		image_indices = sync_tensor_across_gpus(torch.cat(all_image_indices)).cpu().numpy()
		pseudo_labels = sync_tensor_across_gpus(torch.cat(all_pseudo_labels)).cpu().numpy()
		probs = sync_tensor_across_gpus(torch.cat(all_probs)).cpu().numpy()

		dataset.gen_pseudolabel = False
		torch.cuda.empty_cache()

		_ , unique_indices = np.unique(image_indices, return_inverse=True)
		image_indices = image_indices[unique_indices]
		pseudo_labels = pseudo_labels[unique_indices]
		probs = probs[unique_indices]

		new_image_list = []
		nums = []
		ori_nums = []
		for i in range(7):
			inst_mask = (pseudo_labels == i)
			ori_nums.append(int(inst_mask.astype(np.float32).sum()))
			inst_indices = image_indices[inst_mask]
			inst_probs = probs[inst_mask]
			resample_num = int(inst_mask.astype(np.float32).sum() * ratios[i])
			indices = np.argsort(inst_probs)[::-1][:resample_num]
			assert len(indices) == resample_num, [len(indices), resample_num]
			for index in inst_indices[indices]:
				image_name = dataset.original_image_list[index]
				new_image_list.append(image_name)
			nums.append(resample_num)
		new_image_list.sort()
		dataset.update_image_list(new_image_list)

		del image_indices
		del pseudo_labels
		del probs

		if dist.get_rank()==0:
			print(f"############ dataset has length {len(dataset.original_image_list)} --> {len(dataset)} {sum(nums)} ###########")
			for i in range(7):
				print(f"### cls {i+1}  {ori_nums[i]} -> {nums[i]}")


		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		# only deal with first class 0-6 race
		return

	def resample_unlabeled_adapt_ratio(self):
		dataset = self.unlabeled_dataloader.dataset
		n_classifier = len(self.cfg.data.labeled_train.n_classes)
		# create pseudo label
		dataset.gen_pseudolabel = True

		all_image_indices = []
		all_pseudo_labels = []
		all_probs = []

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		for i, unlabeled_batch in enumerate(self.unlabeled_dataloader):
			weak_inputs, image_index = unlabeled_batch
			inputs = weak_inputs.cuda()
			image_index = image_index.cuda()
			probs, _ = self.model(inputs)

			probs = probs[0]
			pseudo_labels = probs.argmax(dim=1)
			probs = torch.max(probs.softmax(dim=1), dim=1)[0]

			torch.cuda.empty_cache()

			all_image_indices.append(image_index)
			all_pseudo_labels.append(pseudo_labels)
			all_probs.append(probs)

		image_indices = sync_tensor_across_gpus(torch.cat(all_image_indices)).cpu().numpy()
		pseudo_labels = sync_tensor_across_gpus(torch.cat(all_pseudo_labels)).cpu().numpy()
		probs = sync_tensor_across_gpus(torch.cat(all_probs)).cpu().numpy()

		dataset.gen_pseudolabel = False
		torch.cuda.empty_cache()

		_, unique_indices = np.unique(image_indices, return_index=True)

		image_indices = image_indices[unique_indices]
		pseudo_labels = pseudo_labels[unique_indices]
		probs = probs[unique_indices]

		new_image_list = []
		cls_nums = []
		for i in range(7):
			inst_mask = (pseudo_labels == i)
			cls_nums.append(np.sum(inst_mask.astype(np.float32)))

		cls_nums = np.array(cls_nums)
		sorted_order = np.argsort(cls_nums)[::-1]
		cls_nums = cls_nums[sorted_order]

		ratios = [ (cls_nums[6-i]/cls_nums[0])**0.5 for i in range(7)]
		rev_order = np.argsort(sorted_order)

		nums = []
		ori_nums = []
		for i in range(7):
			inst_mask = (pseudo_labels == i)
			ori_nums.append(int(inst_mask.astype(np.float32).sum()))
			ratio = ratios[rev_order[i]]
			inst_indices = image_indices[inst_mask]
			inst_probs = probs[inst_mask]
			resample_num = max(int(inst_mask.astype(np.float32).sum() * ratio), 100)
			indices = np.argsort(inst_probs)[::-1][:resample_num]
			# assert len(indices) == resample_num, [len(indices), resample_num]
			for index in inst_indices[indices]:
				assert index < len(dataset.original_image_list), [index, len(dataset.original_image_list), resample_num]
				image_name = dataset.original_image_list[index]
				new_image_list.append(image_name)
			nums.append(resample_num)
		new_image_list.sort()
		dataset.update_image_list(new_image_list)

		del image_indices
		del pseudo_labels
		del probs

		if dist.get_rank()==0:
			print(f"############ dataset has length {len(dataset.original_image_list)} --> {len(dataset)} {sum(nums)} ###########")
			for i in range(7):
				print(f"### cls {i+1}  {ori_nums[i]} -> {nums[i]}")

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		# only deal with first class 0-6 race
		return

	def calculate_pseudo_weights(self):
		dataset = self.unlabeled_dataloader.dataset
		n_classifier = len(self.cfg.data.labeled_train.n_classes)
		# create pseudo label
		dataset.gen_pseudolabel = True

		all_image_indices = []
		all_pseudo_labels = []

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		for i, unlabeled_batch in enumerate(self.unlabeled_dataloader):
			weak_inputs, image_index = unlabeled_batch
			inputs = weak_inputs.cuda()
			image_index = image_index.cuda()
			probs, _ = self.model(inputs)

			probs = probs[0]
			pseudo_labels = probs.argmax(dim=1)

			torch.cuda.empty_cache()

			all_image_indices.append(image_index)
			all_pseudo_labels.append(pseudo_labels)

		image_indices = sync_tensor_across_gpus(torch.cat(all_image_indices)).cpu().numpy()
		pseudo_labels = sync_tensor_across_gpus(torch.cat(all_pseudo_labels)).cpu().numpy()

		dataset.gen_pseudolabel = False
		torch.cuda.empty_cache()

		_, unique_indices = np.unique(image_indices, return_index=True)

		pseudo_labels = pseudo_labels[unique_indices]

		pseudo_weight = np.eye(7)[pseudo_labels] # (B, 7)
		pseudo_cnts = pseudo_weight.sum(axis=0) / pseudo_weight.sum()
		pseudo_weights = 1. / (pseudo_cnts + 1e-6)
		pseudo_weights /= np.sum(pseudo_weights)

		return torch.tensor(pseudo_weights).float().cuda()

	def update_loss_weights(self):
		n_class = self.cfg.model.head.n_classes[0]

		losses = []
		labels = []

		self.model.train()
		mean_loss = []
		with torch.no_grad():
			for i, (inputs, target) in enumerate(self.labeled_dataloader):
				inputs = inputs.cuda()
				target = target.cuda()
				output, loss_dict = self.model(inputs, labels=target)

				loss = loss_dict['cls_1_loss']
				label = target[:, 0]

				losses.append(loss)
				labels.append(label)
			losses = torch.cat(losses, dim=0)
			labels = torch.cat(labels, dim=0)
			losses = sync_tensor_across_gpus(losses).cpu()
			labels = sync_tensor_across_gpus(labels).cpu()

			class_nums = []
			for i in range(n_class):
				class_indices = torch.nonzero(labels==i)
				class_loss = losses[class_indices]
				mean_loss.append(class_loss.mean())
				class_nums.append(len(class_indices))

			cls_loss_weight = torch.tensor(mean_loss, requires_grad=False).cuda()
			cls_loss_weight = cls_loss_weight - cls_loss_weight.min()

			cls_loss_weight = cls_loss_weight / cls_loss_weight.sum()
			cls_loss_weight = cls_loss_weight + 0.0001  # (0.01 as margin)
			self.cls_loss_weight = cls_loss_weight

	def resample_unlabeled_minimum(self):
		dataset = self.unlabeled_dataloader.dataset
		n_classifier = len(self.cfg.data.labeled_train.n_classes)
		# create pseudo label
		dataset.gen_pseudolabel = True

		all_image_indices = []
		all_pseudo_labels = []
		all_probs = []

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		for i, unlabeled_batch in enumerate(self.unlabeled_dataloader):
			weak_inputs, image_index = unlabeled_batch
			inputs = weak_inputs.cuda()
			image_index = image_index.cuda()
			probs, _ = self.model(inputs)

			probs = probs[0]
			pseudo_labels = probs.argmax(dim=1)
			probs = torch.max(probs.softmax(dim=1), dim=1)[0]

			torch.cuda.empty_cache()

			all_image_indices.append(image_index)
			all_pseudo_labels.append(pseudo_labels)
			all_probs.append(probs)

		image_indices = sync_tensor_across_gpus(torch.cat(all_image_indices)).cpu().numpy()
		pseudo_labels = sync_tensor_across_gpus(torch.cat(all_pseudo_labels)).cpu().numpy()
		probs = sync_tensor_across_gpus(torch.cat(all_probs)).cpu().numpy()

		dataset.gen_pseudolabel = False
		torch.cuda.empty_cache()

		_, unique_indices = np.unique(image_indices, return_index=True)

		image_indices = image_indices[unique_indices]
		pseudo_labels = pseudo_labels[unique_indices]
		probs = probs[unique_indices]

		new_image_list = []
		cls_nums = []
		for i in range(7):
			inst_mask = (pseudo_labels == i)
			cls_nums.append(np.sum(inst_mask.astype(np.float32)))

		cls_nums = np.array(cls_nums)
		sorted_order = np.argsort(cls_nums)[::-1]
		cls_nums = cls_nums[sorted_order]

		sample_per_cls = max(int(cls_nums.min()), 100)

		nums = []
		ori_nums = []
		for i in range(7):
			inst_mask = (pseudo_labels == i)
			ori_nums.append(int(inst_mask.astype(np.float32).sum()))
			inst_indices = image_indices[inst_mask]
			inst_probs = probs[inst_mask]
			resample_num = sample_per_cls
			indices = np.argsort(inst_probs)[::-1][:resample_num]
			# assert len(indices) == resample_num, [len(indices), resample_num]
			for index in inst_indices[indices]:
				assert index < len(dataset.original_image_list), [index, len(dataset.original_image_list), resample_num]
				image_name = dataset.original_image_list[index]
				new_image_list.append(image_name)
			nums.append(resample_num)
		new_image_list.sort()
		dataset.update_image_list(new_image_list)

		del image_indices
		del pseudo_labels
		del probs

		if dist.get_rank()==0:
			print(f"############ dataset has length {len(dataset.original_image_list)} --> {len(dataset)} {sum(nums)} ###########")
			for i in range(7):
				print(f"### cls {i+1}  {ori_nums[i]} -> {nums[i]}")

		unlabeled_sampler = DistributedSampler(dataset, shuffle=True)
		del self.unlabeled_dataloader
		torch.cuda.empty_cache()
		self.unlabeled_dataloader = DataLoader(dataset,
											   sampler=unlabeled_sampler,
											   batch_size=self.cfg.data.unlabeled_train.samples_per_gpu,
											   num_workers=self.cfg.data.unlabeled_train.workers_per_gpu,
											   drop_last=False)

		# only deal with first class 0-6 race
		return

	def run(self):
		n_classifier = len(self.cfg.data.labeled_train.n_classes)

		max_epochs = self.cfg.trainer.max_epochs

		init_epoch = self.epoch + 1

		best_top1 = 0.
		best_epoch = 0

		train_time = AverageMeter('Time', ':6.3f')

		end = time.time()

		self.model.train()

		for self.epoch in range(init_epoch, max_epochs+1):

			self.lr_scheduler.step()
			self.train()
			
			save_best = False
			if self.epoch % self.cfg.trainer.val_freq == 0:
				top1s, cls_accs, cls_precs = self.validate()
				if top1s[0] > best_top1:
					best_top1 = top1s[0]
					best_epoch = self.epoch
					save_best = True
				output = "Eval Results: \n"
				for i in range(n_classifier):
					output += f'attr_{i+1} Prec@1 {top1s[i]:.3f}'
					if i == 0:
						output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f} \n'
					out_cls_acc = 'Eval Class Accuracy: \n%s' % (
						(np.array2string(cls_accs[i], separator='\n', formatter={'float_kind': lambda x: "%.3f" % x})))
					output += out_cls_acc
					output += "\n"
				self.logger.info(output)

			train_time.update(time.time() - end)
			end = time.time()
			
			if self.epoch % self.cfg.trainer.save_freq == 0:
				self.save_checkpoint(latest=True, best=save_best)

			eta = train_time.avg * (max_epochs - self.epoch)
			format_eta = str(timedelta(seconds=eta))
			self.logger.info(f'######### ETA {format_eta} ##########')

		self.logger.info("########## Training Finished ###########")
		if dist.get_rank() == 0:
			self.logger.info("########## Evaluating last epoch ###########")
			top1s, cls_recalls, cls_precs = self.test(bias_validate=False)
			output = "Eval Results: \n"
			
			for i in range(n_classifier):
				output += f'attr_{i + 1} Prec@1 {top1s[i]:.3f} '
				if i == 0:
					output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f}'
				out_cls_recalls = 'Eval Class Recalls: \n%s' % (
					(np.array2string(cls_recalls[i], separator='\n', formatter={'float_kind': lambda x: "%.3f" % x})))
				output += out_cls_recalls
				output += "\n"
				
				out_cls_precs = 'Eval Class Precs: \n%s' % (
					(np.array2string(cls_precs[i], separator='\n', formatter={'float_kind': lambda x: "%.3f" % x})))
				output += out_cls_precs
				output += "\n"
				
				self.logger.info(output)
			
			self.logger.info("########## Evaluating best epoch ###########")
			self.load_from(f"{self.cfg.work_dir}/best_model.pth")
			top1s, cls_recalls, cls_precs = self.test(bias_validate=False)
			output = "Eval Results: \n"
			
			for i in range(n_classifier):
				output += f'attr_{i + 1} Prec@1 {top1s[i]:.3f} '
				if i == 0:
					output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f}'
				out_cls_recalls = 'Eval Class Recalls: \n%s' % (
					(np.array2string(cls_recalls[i], separator='\n', formatter={'float_kind': lambda x: "%.3f" % x})))
				output += out_cls_recalls
				output += "\n"
				
				out_cls_precs = 'Eval Class Precs: \n%s' % (
					(np.array2string(cls_precs[i], separator='\n', formatter={'float_kind': lambda x: "%.3f" % x})))
				output += out_cls_precs
				output += "\n"
				
				self.logger.info(output)

	def train(self):
		if self.epoch % 1 == 0:
			if self.cfg.trainer.resample is None:
				pass
			elif self.cfg.trainer.resample == 'fixed_ratio':
				self.model.eval()
				with torch.no_grad():
					self.resample_unlabeled_fixed_ratio()
			elif self.cfg.trainer.resample == 'adapt_ratio':
				self.model.eval()
				with torch.no_grad():
					self.resample_unlabeled_adapt_ratio()
			elif self.cfg.trainer.resample == 'minimum_cls':
				self.model.eval()
				with torch.no_grad():
					self.resample_unlabeled_minimum()

		if self.cfg.trainer.adapt_pseudo_weights:
			self.model.eval()
			with torch.no_grad():
				pseudo_weights = self.calculate_pseudo_weights()

		if self.cfg.trainer.use_weight:
			# epoch update
			# if self.epoch >= 0:
			#     self.update_loss_weights()
			cls_weight_str = ', '.join([f"{i:0.2f}" for i in self.cls_loss_weight.cpu().numpy()])
			self.logger.info(f" loss weight : {cls_weight_str}")

		self.model.train()

		batch_time = AverageMeter('Time', ':6.3f')
		data_time = AverageMeter('Data', ':6.3f')
		losses = AverageMeter('Loss', ':.4e')
		top1 = AverageMeter('Acc@1', ':6.2f')

		n_class = self.cfg.model.head.n_classes[0]
		n_classifier = len(self.cfg.data.labeled_train.n_classes)
		labeled_losses = [AverageMeter('Loss', ':.4e') for i in range(n_classifier)]
		unlabeled_losses = [AverageMeter('Loss', ':.4e') for i in range(n_classifier)]

		self.model.train()

		end = time.time()

		threshold = 0 if self.epoch < 10 else 0.

		labeled_iter = iter(self.labeled_dataloader)
		unlabeled_iter = iter(self.unlabeled_dataloader)

		stop_on_unlabeled = len(self.unlabeled_dataloader) > len(self.labeled_dataloader)

		n_iter = len(self.unlabeled_dataloader) if len(self.unlabeled_dataloader) > len(self.labeled_dataloader) else len(self.labeled_dataloader)
		# for i, unlabeled_batch in enumerate(self.unlabeled_dataloader):
		i_iter = 0
		
		avg_loss = 0.
		races_added = [0, 0, 0, 0]
		while True:
			try:
				labeled_batch = next(labeled_iter)
			except:
				if not stop_on_unlabeled:
					break
				labeled_iter = iter(self.labeled_dataloader)
				labeled_batch = next(labeled_iter)
			try:
				unlabeled_batch = next(unlabeled_iter)
			except:
				if stop_on_unlabeled:
					break
				unlabeled_iter = iter(self.unlabeled_dataloader)
				unlabeled_batch = next(unlabeled_iter)

			weak_unlabeled_inputs, strong_unlabeled_inputs = unlabeled_batch

			data_time.update(time.time() - end)

			labeled_inputs = labeled_batch[0].cuda()

			target = labeled_batch[1]

			labels = target
			if self.cfg.model.loss == 'kld':
				labels = target[:, 0]
				labels = torch.eye(self.cfg.data.labeled_train.n_classes[0])[labels].unsqueeze(1).contiguous()
			target = target.cuda()
			labels = labels.cuda()

			output, loss_dict, pseudo_labels = self.model(labeled_inputs, weak_unlabeled_inputs, strong_unlabeled_inputs, labels,
										   threshold=threshold, label_weights=self.cls_loss_weight, unlabel_weights=self.unlabeled_cls_loss_weight)
			pseudo_labels = pseudo_labels[:, 0]
			pseudo_labels = sync_tensor_across_gpus(pseudo_labels)
			for i in range(4):
				races_added[i] += torch.sum((pseudo_labels == i).float()).cpu().numpy().item()

			for key in loss_dict:
				if 'ssl' in key:
					loss_dict[key] *= self.cfg.trainer.ssl_weight
				if 'cls' in key:
					loss_dict[key] *= self.cfg.trainer.cls_weight

			loss = sum([t.mean() for t in loss_dict.values()])

			# update loss weight per iteration using momentum update
			# it has latency, works not as good as epoch instant update
			# if self.epoch >= 0:
			# 	with torch.no_grad():
			# 		# compute cls loss weight
			# 		mean_loss = []
			# 		stat_loss = loss_dict['cls_1_loss']
			# 		stat_label = target[:, 0].contiguous()
			#
			# 		stat_losses = sync_tensor_across_gpus(stat_loss).cpu()
			# 		stat_labels = sync_tensor_across_gpus(stat_label).cpu()
			#
			# 		for i in range(n_class):
			# 			class_indices = torch.nonzero(stat_labels == i)
			# 			if len(class_indices) == 0:
			# 				mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
			# 			else:
			# 				class_loss = stat_losses[class_indices]
			# 				mean_loss.append(class_loss.mean() / self.cls_loss_weight[i])
			# 		cls_loss_weight = torch.tensor(mean_loss, requires_grad=False).cuda()
			# 		cls_loss_weight = cls_loss_weight - cls_loss_weight.min()
			# 		combined_loss_weight = cls_loss_weight / cls_loss_weight.sum()
			#
			# 		# compute ssl loss weight
			# 		# mean_loss = []
			# 		# stat_loss = loss_dict['ssl_1_loss']
			# 		# stat_label = pseudo_labels[:, 0].contiguous()
			# 		#
			# 		# stat_losses = sync_tensor_across_gpus(stat_loss).cpu()
			# 		# stat_labels = sync_tensor_across_gpus(stat_label).cpu()
			# 		#
			# 		# for i in range(n_class):
			# 		#     class_indices = torch.nonzero(stat_labels == i)
			# 		#     if len(class_indices) == 0:
			# 		#         mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
			# 		#     else:
			# 		#         class_loss = stat_losses[class_indices]
			# 		#         mean_loss.append(class_loss.mean() / self.cls_loss_weight[i])
			# 		# ssl_loss_weight = torch.tensor(mean_loss, requires_grad=False).cuda()
			# 		# ssl_loss_weight = ssl_loss_weight - ssl_loss_weight.min()
			# 		# ssl_loss_weight = ssl_loss_weight / ssl_loss_weight.sum()
			#
			# 		# combined_loss_weight = 0.1 * cls_loss_weight + 0.9 * ssl_loss_weight
			#
			# 		combined_loss_weight = combined_loss_weight + 0.0001  # (0.0001 as margin)
			#
			# 		self.cls_loss_weight = 0.9 * self.cls_loss_weight + 0.1 * combined_loss_weight

			acc1 = accuracy(output[0], target[:, 0], topk=(1, ))[0]
			losses.update(loss.item(), labeled_inputs.size(0) + weak_unlabeled_inputs.size(0))
			top1.update(acc1[0], labeled_inputs.size(0) + weak_unlabeled_inputs.size(0))

			for j in range(n_classifier):
				labeled_losses[j].update(loss_dict[f'cls_{j+1}_loss'].mean().item(), labeled_inputs.size(0))
				unlabeled_losses[j].update(loss_dict[f"ssl_{j+1}_loss"].mean().item(), weak_unlabeled_inputs.size(0))

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
			self.optimizer.step()

			self.model.module.update_ema()

			torch.cuda.empty_cache()

			batch_time.update(time.time() - end)
			end = time.time()

			if i_iter % self.cfg.trainer.print_freq == 0:
				output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
						  'Time {batch_time.val:.3f} \t'
						  'Data ({data_time.val:.3f})\t'
						  'Loss {loss.val:.4f} \t'
						  'Prec@1 {top1.val:.3f} \t'.format(
					self.epoch, i_iter, n_iter, batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1,
					lr=self.optimizer.param_groups[-1]['lr']))
				for j in range(n_classifier):
					output += f"cls_{j+1}/ssl_{j+1} = {labeled_losses[j].val:.4f}/{unlabeled_losses[j].val:.4f}\t"
				self.logger.info(output)

			i_iter += 1

		if dist.get_rank()==0:
			if 'use_covariance_matrix' in self.cfg.model and self.cfg.model.use_covariance_matrix:
				for t in self.model.module.covariance_matrix:
					print(t)
			self.writer.add_scalar('loss/train_all', losses.avg, self.epoch)
			self.writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
			self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], self.epoch)
			for i in range(n_classifier):
				self.writer.add_scalar(f'loss/train_cls_{i+1}', labeled_losses[i].avg, self.epoch)
				self.writer.add_scalar(f'loss/train_ssl_{i+1}', unlabeled_losses[i].avg, self.epoch)
		
		self.logger.info(races_added)
		self.logger.info(f'$$$$ epoch avg loss labeled: {labeled_losses[0].avg}  unlabeled: {unlabeled_losses[0].avg} $$$')

	def validate(self, record=True, bias_validate=False, save_probs=False, save_embeddings=False):
		batch_time = AverageMeter('Time', ':6.3f')

		n_classes = self.cfg.data.labeled_train.n_classes
		n_classifier = len(n_classes)
		top1s = []
		for i in range(n_classifier):
			top1s.append(AverageMeter('Acc@1', ':6.2f'))

		# switch to evaluate mode
		self.model.eval()
		all_preds = []
		all_targets = []
		all_probs = []
		for i in range(n_classifier):
			all_preds.append([])
			all_targets.append([])
			all_probs.append([])

		if save_embeddings:
			embeddings_list = []
			targets_list = []

		with torch.no_grad():
			end = time.time()
			for i, (inputs, target, weights) in enumerate(self.val_loader):
				inputs = inputs.cuda()
				target = target.cuda()

				output, _ = self.model.module(inputs)

				if save_embeddings:
					embeddings = self.model.module.get_embeddings(inputs)
					embeddings_list.append(embeddings.cpu().numpy())
					targets_list.append(target.cpu().numpy())

				for j in range(n_classifier):
					acc1 = accuracy(output[j], target[:, j], topk=(1, ))[0]
					top1s[j].update(acc1[0], inputs.size(0))

					all_probs[j].extend(output[j].cpu().numpy())
					_, pred = torch.max(output[j], 1)
					all_preds[j].extend(pred.cpu().numpy())
					all_targets[j].extend(target[:, j].cpu().numpy())

				batch_time.update(time.time() - end)
				end = time.time()

			cls_accs = []
			cls_precs = []
			for i in range(n_classifier):
				if i==0:
					# all_preds[i][all_preds[i] == 3] = 2
					# all_preds[i][all_preds[i] == 4] = 3
					# all_preds[i][all_preds[i] == 5] = 0
					cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(4)).astype(float)
					self.logger.info(f"\n{cf}")
				else:
					# cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(n_classes[i])).astype(float)
					cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(n_classes[i])).astype(float)
				cls_cnt = 1e-6 + cf.sum(axis=1)
				cls_hit = np.diag(cf)
				cls_acc = cls_hit / cls_cnt
				cls_accs.append(cls_acc)
				cls_precs.append(cls_hit / (1e-6+cf.sum(axis=0)))

				if i==0:
					non_white_hit = cls_hit[1:].sum()
					non_white_targets = cls_cnt[1:].sum()
					non_white_accs = non_white_hit / non_white_targets
					self.logger.info(f'white_acc {cls_acc[0]} non_white_acc {non_white_accs}')

			if record and dist.get_rank() == 0:
				for i in range(n_classifier):
					self.writer.add_scalar(f'Eval/acc_top1_{i+1}', top1s[i].avg, self.epoch)
					self.writer.add_scalars(f'Eval/acc_cls_{i+1}', {str(i): x for i, x in enumerate(cls_accs[i])},
											self.epoch)

			if save_probs:
				all_probs_np = np.array(all_probs[0])
				all_targets_np = np.array(all_targets[0])
				np.save('fakemake_minimum-cls_on_fairface_val_probs', all_probs_np)
				np.save('fakemake_minimum-cls_on_fairface_val_targets', all_targets_np)

			if save_embeddings:
				embeddings = np.concatenate(embeddings_list)
				targets = np.concatenate(targets_list)
				np.save('fixmatch_fairface-val_embeddings', embeddings)
				np.save('fixmatch_fairface-val_labels', targets)

			if bias_validate:
				n_classes_sum = sum(n_classes)
				bias_accs = np.zeros_like((n_classes_sum, n_classes_sum), dtype=np.float32)

				row_index = 0
				for i in range(n_classifier):
					n_class = n_classes[i]
					for j in range(n_class):
						target_mask = all_targets[i] == j
						col_index = 0
						for k in range(n_classifier):
							if i == k:
								col_index += n_classes[k]
								continue
							all_target = all_targets[k][target_mask]
							all_pred = all_preds[k][target_mask]
							cf = confusion_matrix(all_target, all_pred).astype(float)
							cls_cnt = cf.sum(axis=1)
							cls_hit = np.diag(cf)
							cls_acc = cls_hit / cls_cnt
							bias_accs[row_index][col_index:col_index+n_classes[k]] = cls_acc
							col_index += n_classes[k]
						row_index += 1
				return [top1.avg for top1 in top1s],  cls_accs, bias_accs

		return [top1.avg for top1 in top1s],  cls_accs, cls_precs

	def test(self, record=True, bias_validate=False, save_probs=False, save_embeddings=False):
		batch_time = AverageMeter('Time', ':6.3f')

		n_classes = self.cfg.data.labeled_train.n_classes
		n_classifier = len(n_classes)
		top1s = []
		for i in range(n_classifier):
			top1s.append(AverageMeter('Acc@1', ':6.2f'))

		# switch to evaluate mode
		self.model.eval()
		all_preds = []
		all_targets = []
		all_probs = []
		for i in range(n_classifier):
			all_preds.append([])
			all_targets.append([])
			all_probs.append([])

		if save_embeddings:
			embeddings_list = []
			targets_list = []

		with torch.no_grad():
			end = time.time()
			for i, (inputs, target, weights) in enumerate(self.test_loader):
				inputs = inputs.cuda()
				target = target.cuda()

				output, _ = self.model.module(inputs)

				if save_embeddings:
					embeddings = self.model.module.get_embeddings(inputs)
					embeddings_list.append(embeddings.cpu().numpy())
					targets_list.append(target.cpu().numpy())

				for j in range(n_classifier):
					acc1 = accuracy(output[j], target[:, j], topk=(1, ))[0]
					top1s[j].update(acc1[0], inputs.size(0))

					all_probs[j].extend(output[j].cpu().numpy())
					_, pred = torch.max(output[j], 1)
					all_preds[j].extend(pred.cpu().numpy())
					all_targets[j].extend(target[:, j].cpu().numpy())

				batch_time.update(time.time() - end)
				end = time.time()

			cls_accs = []
			cls_precs = []
			for i in range(n_classifier):
				if i==0:
					# all_preds[i][all_preds[i] == 3] = 2
					# all_preds[i][all_preds[i] == 4] = 3
					# all_preds[i][all_preds[i] == 5] = 0
					cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(4)).astype(float)
					self.logger.info(f"\n{cf}")
				else:
					# cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(n_classes[i])).astype(float)
					cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(n_classes[i])).astype(float)
				cls_cnt = 1e-6 + cf.sum(axis=1)
				cls_hit = np.diag(cf)
				cls_acc = cls_hit / cls_cnt
				cls_accs.append(cls_acc)
				cls_precs.append(cls_hit / (1e-6+cf.sum(axis=0)))

				if i==0:
					non_white_hit = cls_hit[1:].sum()
					non_white_targets = cls_cnt[1:].sum()
					non_white_accs = non_white_hit / non_white_targets
					self.logger.info(f'white_acc {cls_acc[0]} non_white_acc {non_white_accs}')

			if record and dist.get_rank() == 0:
				for i in range(n_classifier):
					self.writer.add_scalar(f'Eval/acc_top1_{i+1}', top1s[i].avg, self.epoch)
					self.writer.add_scalars(f'Eval/acc_cls_{i+1}', {str(i): x for i, x in enumerate(cls_accs[i])},
											self.epoch)

			if save_probs:
				all_probs_np = np.array(all_probs[0])
				all_targets_np = np.array(all_targets[0])
				np.save('fakemake_minimum-cls_on_fairface_val_probs', all_probs_np)
				np.save('fakemake_minimum-cls_on_fairface_val_targets', all_targets_np)

			if save_embeddings:
				embeddings = np.concatenate(embeddings_list)
				targets = np.concatenate(targets_list)
				np.save('fixmatch_fairface-val_embeddings', embeddings)
				np.save('fixmatch_fairface-val_labels', targets)

			if bias_validate:
				n_classes_sum = sum(n_classes)
				bias_accs = np.zeros_like((n_classes_sum, n_classes_sum), dtype=np.float32)

				row_index = 0
				for i in range(n_classifier):
					n_class = n_classes[i]
					for j in range(n_class):
						target_mask = all_targets[i] == j
						col_index = 0
						for k in range(n_classifier):
							if i == k:
								col_index += n_classes[k]
								continue
							all_target = all_targets[k][target_mask]
							all_pred = all_preds[k][target_mask]
							cf = confusion_matrix(all_target, all_pred).astype(float)
							cls_cnt = cf.sum(axis=1)
							cls_hit = np.diag(cf)
							cls_acc = cls_hit / cls_cnt
							bias_accs[row_index][col_index:col_index+n_classes[k]] = cls_acc
							col_index += n_classes[k]
						row_index += 1
				return [top1.avg for top1 in top1s],  cls_accs, bias_accs

		return [top1.avg for top1 in top1s],  cls_accs, cls_precs



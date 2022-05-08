# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings
import time
from collections import OrderedDict
import shutil
from datetime import timedelta
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from balface.models import build_model
from balface.datasets import build_dataset
from balface.utils import get_root_logger, AverageMeter, accuracy, sync_tensor_across_gpus
from warmup_scheduler import GradualWarmupScheduler


def build_optimizer(model, opt_cfg):
    assert opt_cfg.name in ['sgd', 'adam', 'adamw']
    lr = opt_cfg.lr
    weight_decay = opt_cfg.weight_decay
    if opt_cfg.name == 'sgd':
        momentum = opt_cfg.momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_cfg.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_cfg.name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer

# def build_lr_scheduler(scheduler_cfg, optimizer, last_epoch):
#     assert scheduler_cfg.name in ['multisteplrwwarmup', 'multisteplr']
#     elif scheduler_cfg.name == 'multisteplr':
#         milestones = scheduler_cfg.milestones
#         return MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=last_epoch)

class CifarTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.logger = cfg.logger
        if dist.get_rank()==0:
            self.writer = SummaryWriter(f"{self.cfg.work_dir}/tfb")

        self.epoch = -1

        model = build_model(cfg.model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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

        # self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, self.optimizer, last_epoch=self.epoch)

        train_dataset = build_dataset(cfg.data.train)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # prepare data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.train.samples_per_gpu,
            num_workers=cfg.data.train.workers_per_gpu,
            sampler=train_sampler,
            drop_last=False
        )
        val_dataset = build_dataset(cfg.data.val)
        # prepare data loaders
        # single batch data loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.val.samples_per_gpu,
            num_workers=cfg.data.val.workers_per_gpu,
            drop_last=False
        )

        n_class = self.cfg.model.head.n_class
        self.cls_loss_weight = torch.ones(n_class, requires_grad=False).cuda()
        
        cls_num_list = train_dataset.get_cls_num_list()
        self.sample_ratio = np.array(cls_num_list, dtype=np.float32)

        # self.cls_loss_weight[0] /= 10.
        # self.cls_loss_weight /= self.cls_loss_weight.sum()

    def resume(self, resume_from):
        save_content = torch.load(resume_from)
        epoch = save_content['epoch']
        statedict = save_content['state_dict']
        opt_state = save_content['opt_state']

        self.model.module.load_state_dict(statedict)
        self.optimizer.load_state_dict(opt_state)
        self.epoch = epoch

    def load_backbone(self, load_from):
        save_content = torch.load(load_from)
        statedict = save_content['state_dict']
        backbone_state_dict = OrderedDict()
        for key in statedict.keys():
            if 'backbone' in key:
                backbone_state_dict[key] = statedict[key]
        self.model.module.load_state_dict(backbone_state_dict, strict=False)

    def save_checkpoint(self, latest=True):
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

    def run(self):
        n_classifier = 1

        max_epochs = self.cfg.trainer.max_epochs

        init_epoch = self.epoch + 1

        best_top1 = 0.
        best_epoch = 0

        train_time = AverageMeter('Time', ':6.3f')

        end = time.time()
        for self.epoch in range(init_epoch, max_epochs+1):
            self.train()
            # self.lr_scheduler.step()
            self.adjust_lr()

            if self.epoch % self.cfg.trainer.save_freq == 0:
                self.save_checkpoint(latest=True)

            if self.epoch % self.cfg.trainer.val_freq == 0:
                top1, cls_acc, cls_prec = self.validate()
                if top1 > best_top1:
                    best_top1 = top1
                    best_epoch = self.epoch
                output = "Eval Results: \n"
                output += f'attr Prec@1 {top1:.3f}'
                output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f}\n'
                out_cls_acc = 'Eval Class Recalls: \n%s' % (
                    (np.array2string(cls_acc, separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
                output += out_cls_acc
                output += "\n"
                output_cls_acc_diff = f'class acc diff: {cls_acc.max() - cls_acc.min()}'
                output += output_cls_acc_diff
                output += "\n"

                # out_cls_acc = 'Eval Class Precs: \n%s' % (
                #     (np.array2string(cls_precs[i], separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
                # output += out_cls_acc
                # output += "\n"

                self.logger.info(output)

            train_time.update(time.time() - end)
            end = time.time()

            eta = train_time.avg * (max_epochs - self.epoch)
            format_eta = str(timedelta(seconds=eta))
            self.logger.info(f'######### ETA {format_eta} ##########')

        self.logger.info("########## Training Finished ###########")
        if dist.get_rank() == 0:
            top1, cls_acc, cls_prec = self.validate()
            if top1 > best_top1:
                best_top1 = top1
                best_epoch = self.epoch
            output = "Eval Results: \n"
            output += f'attr Prec@1 {top1:.3f}'
            output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f}\n'
            out_cls_acc = 'Eval Class Recalls: \n%s' % (
                (np.array2string(cls_acc, separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
            output += out_cls_acc
            output += "\n"
            output_cls_acc_diff = f'class acc diff: {cls_acc.max() - cls_acc.min()}'
            output += output_cls_acc_diff
            output += "\n"

            # out_cls_acc = 'Eval Class Precs: \n%s' % (
            #     (np.array2string(cls_precs[i], separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
            # output += out_cls_acc
            # output += "\n"

            self.logger.info(output)


    def adjust_lr(self):
        if self.cfg.trainer.loss_weight == 'none':
            if self.epoch <= 5:
                lr = self.cfg.optimizer.lr * self.epoch / 5
            elif self.epoch > 180:
                lr = self.cfg.optimizer.lr * 0.0001
            elif self.epoch > 160:
                lr = self.cfg.optimizer.lr * 0.01
            else:
                lr = self.cfg.optimizer.lr
        else:
            if self.epoch <= 5:
                lr = self.cfg.optimizer.lr * self.epoch / 5
            elif self.epoch > 180:
                lr = self.cfg.optimizer.lr * 0.0001
            elif self.epoch > 160:
                lr = self.cfg.optimizer.lr * 0.01
            else:
                lr = self.cfg.optimizer.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_loss_weight_v1(self):
        n_class = self.cfg.model.head.n_class

        probs = []
        labels = []

        self.model.train()
        mean_prob = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                output, loss_dict = self.model(inputs, target)

                assert len(output) == 1
                output = output[0]
                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target.contiguous(), num_classes=n_class).float().cuda()), dim=1)
                prob = prob
                label = target

                probs.append(prob)
                labels.append(label)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())

                class_nums.append(len(class_indices))

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * self.cfg.trainer.expected_ratio

            sample_ratio = torch.tensor(self.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_error_rate**2 * (1-cls_error_rate))) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            cls_loss_weight = cls_loss_weight.clamp(0., 5.)
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weight_v2(self):
        n_class = self.cfg.model.head.n_class

        probs = []
        vars = []
        labels = []

        self.model.train()
        mean_prob = []
        mean_var = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                output, loss_dict = self.model(inputs, target)
                assert len(output) == 1
                output = output[0]

                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target.contiguous(), num_classes=n_class).float().cuda()), dim=1)
                var = prob * (1-prob)
                label = target

                probs.append(prob)
                labels.append(label)
                vars.append(var)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            vars = torch.cat(vars, dim=0)
            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()
            vars = sync_tensor_across_gpus(vars).cpu()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())
                class_var = vars[class_indices]
                mean_var.append(class_var.mean())

                class_nums.append(len(class_indices))
            # print(mean_prob)

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            cls_var_rate = torch.tensor(mean_var, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * self.cfg.trainer.expected_ratio

            sample_ratio = torch.tensor(self.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_error_rate * cls_var_rate)) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class
            cls_loss_weight = cls_loss_weight.clamp(0., 10.)
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weight_v3(self):
        n_class = self.cfg.model.head.n_class

        probs = []
        vars = []
        labels = []
        norms = []

        self.model.train()
        mean_prob = []
        mean_var = []
        mean_norm = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                output, loss_dict = self.model(inputs, target)
                assert len(output) == 1
                output = output[0]
                embeddings = self.model.module.get_embeddings(inputs)
                embeddings_norms = torch.linalg.norm(embeddings, dim=1)

                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target.contiguous(), num_classes=n_class).float().cuda()), dim=1)
                var = prob * (1-prob)
                embeddings_norms = (1-prob) * embeddings_norms

                label = target

                probs.append(prob)
                labels.append(label)
                vars.append(var)
                norms.append(embeddings_norms)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            vars = torch.cat(vars, dim=0)
            embeddings_norms = torch.cat(norms, dim=0)

            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()
            vars = sync_tensor_across_gpus(vars).cpu()
            embeddings_norms = sync_tensor_across_gpus(embeddings_norms).cpu()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())
                class_var = vars[class_indices]
                mean_var.append(class_var.mean())
                mean_norm.append(embeddings_norms[class_indices].mean())

                class_nums.append(len(class_indices))

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            cls_var_rate = torch.tensor(mean_var, requires_grad=False).cuda()
            cls_embeddings_rate = torch.tensor(mean_norm, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * self.cfg.trainer.expected_ratio

            sample_ratio = torch.tensor(self.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_embeddings_rate * cls_var_rate)) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            cls_loss_weight = cls_loss_weight.clamp(0., 10.)

            self.cls_loss_weight = cls_loss_weight

    def train(self):
        # per epoch loss weight update

        if self.cfg.model.head.loss =='ib-ce':
            self.model.module.head.loss = 'ce'

        if self.cfg.trainer.loss_weight!='none' and self.epoch >= self.cfg.trainer.meta_epoch:
            self.model.module.head.loss = self.cfg.model.head.loss

            if self.cfg.trainer.loss_weight=='v1':
                self.update_loss_weight_v1()
            elif self.cfg.trainer.loss_weight=='v2':
                self.update_loss_weight_v2()
            elif self.cfg.trainer.loss_weight=='v3':
                self.update_loss_weight_v3()
            elif self.cfg.trainer.loss_weight=='none':
                pass
            else:
                raise Exception()

        cls_weight_str = ', '.join([f"{i:0.4f}" for i in self.cls_loss_weight.cpu().numpy()])
        self.logger.info(f" loss weight : {cls_weight_str}")

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        labeled_losses = AverageMeter('Loss', ':.4e')

        self.model.train()

        end = time.time()
        for i, (inputs, target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            target = target.cuda()
            output, loss_dict = self.model(inputs, target, self.cls_loss_weight)

            loss = sum([t.mean() for t in loss_dict.values()])

            acc1 = accuracy(output[0], target, topk=(1, ))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))

            labeled_losses.update(loss_dict[f'cls_loss'].mean().item(), inputs.size(0))


            # update loss weight per iteration using momentum update
            # it has latency, works not as good as epoch instant update
            # if self.epoch >= 10:
            #     with torch.no_grad():
            #         mean_loss = []
            #         stat_loss = loss_dict['cls_1_loss']
            #         stat_label = target[:, 0].contiguous()
            #
            #         stat_losses = sync_tensor_across_gpus(stat_loss).cpu()
            #         stat_labels = sync_tensor_across_gpus(stat_label).cpu()
            #
            #         for ii in range(n_class):
            #             class_indices = torch.nonzero(stat_labels == ii)
            #             if len(class_indices) == 0:
            #                 mean_loss.append(-np.log(0.05) / self.cls_loss_weight[ii])
            #             else:
            #                 class_loss = stat_losses[class_indices]
            #                 mean_loss.append(class_loss.mean() / self.cls_loss_weight[ii])
            #         cls_loss_weight = torch.tensor(mean_loss, requires_grad=False).cuda()
            #         cls_loss_weight = cls_loss_weight - cls_loss_weight.min()
            #
            #         cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() + 1e-4
            #         self.cls_loss_weight = 0.9 * self.cls_loss_weight + 0.1 * cls_loss_weight
            #         self.cls_loss_weight = self.cls_loss_weight / self.cls_loss_weight.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if self.cfg.model.head.norm_weights:
            #     self.model.module.head.norm_classifier_weights()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.cfg.trainer.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} \t'
                          'Data ({data_time.val:.3f})\t'
                          'Loss {loss.val:.4f} \t'
                          'Prec@1 {top1.val:.3f} \t'.format(
                    self.epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1,
                    lr=self.optimizer.param_groups[-1]['lr']))
                output += f"cls_loss = {labeled_losses.val:.4f}\t"
                self.logger.info(output)

        if dist.get_rank() == 0:
            self.writer.add_scalar('loss/train_all', losses.avg, self.epoch)
            self.writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], self.epoch)
            self.writer.add_scalar(f'loss/train_cls', labeled_losses.avg, self.epoch)

    def validate(self, record=True, save_probs=False, save_embeddings=False):
        batch_time = AverageMeter('Time', ':6.3f')
        n_class = self.cfg.data.train.n_class

        top1 = AverageMeter('Acc@1', ':6.2f')

        # switch to evaluate mode
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        if save_embeddings:
            embeddings_list = []
            targets_list = []

        with torch.no_grad():
            end = time.time()
            for i, (inputs, target) in enumerate(self.val_loader):
                inputs = inputs.cuda()
                target = target.cuda()

                output, _ = self.model.module(inputs)

                if save_embeddings:
                    embeddings = self.model.module.get_embeddings(inputs)
                    embeddings_list.append(embeddings.cpu().numpy())
                    targets_list.append(target.cpu().numpy())

                acc1 = accuracy(output[0], target, topk=(1, ))[0]
                top1.update(acc1.detach().cpu().item(), inputs.size(0))

                all_probs.extend(output[0].cpu().numpy())
                _, pred = torch.max(output[0], 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                batch_time.update(time.time() - end)
                end = time.time()

            cf = confusion_matrix(all_targets, all_preds, labels=range(n_class)).astype(float)
            cls_cnt = 1e-6 + cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            cls_prec = cls_hit / (1e-6+cf.sum(axis=0))

            if record and dist.get_rank()==0:
                self.writer.add_scalar(f'Eval/acc_top1', top1.avg, self.epoch)
                self.writer.add_scalars(f'Eval/acc_cls', {str(t): x for t, x in enumerate(cls_acc)}, self.epoch)

            if save_probs:
                all_probs_np = np.array(all_probs[0])
                all_targets_np = np.array(all_targets[0])
                np.save('fairface_train_probs', all_probs_np)
                np.save('fairface_train_targets', all_targets_np)

            if save_embeddings:
                embeddings = np.concatenate(embeddings_list)
                targets = np.concatenate(targets_list)
                np.save('ssp_fairface-val_embeddings', embeddings)
                np.save('ssp_fairface-val_labels', targets)

            return top1.avg, cls_acc, cls_prec

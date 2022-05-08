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

class RSCTrainer:
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

        self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, self.optimizer, last_epoch=self.epoch)

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

        n_class = self.cfg.model.head.n_classes[0]
        self.cls_loss_weight = torch.ones(n_class, requires_grad=False).cuda()
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
        n_classifier = len(self.cfg.data.train.n_classes)

        max_epochs = self.cfg.trainer.max_epochs

        init_epoch = self.epoch + 1

        best_top1 = 0.
        best_epoch = 0

        train_time = AverageMeter('Time', ':6.3f')

        end = time.time()
        for self.epoch in range(init_epoch, max_epochs+1):
            self.train()
            self.lr_scheduler.step()

            if self.epoch % self.cfg.trainer.save_freq == 0:
                self.save_checkpoint(latest=True)

            if self.epoch % self.cfg.trainer.val_freq == 0:
                top1s, cls_accs, cls_precs = self.validate()
                if top1s[0] > best_top1:
                    best_top1 = top1s[0]
                    best_epoch = self.epoch
                output = "Eval Results: \n"
                for i in range(n_classifier):
                    output += f'attr_{i + 1} Prec@1 {top1s[i]:.3f}'
                    if i == 0:
                        output += f'; Best Prec@1 {best_epoch}/{best_top1:.3f}\n'
                    out_cls_acc = 'Eval Class Recalls: \n%s' % (
                        (np.array2string(cls_accs[i], separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
                    output += out_cls_acc
                    output += "\n"

                    out_cls_acc = 'Eval Class Precs: \n%s' % (
                        (np.array2string(cls_precs[i], separator='\t', formatter={'float_kind': lambda x: "%.3f" % x})))
                    output += out_cls_acc
                    output += "\n"

                self.logger.info(output)

            train_time.update(time.time() - end)
            end = time.time()

            eta = train_time.avg * (max_epochs - self.epoch)
            format_eta = str(timedelta(seconds=eta))
            self.logger.info(f'######### ETA {format_eta} ##########')

        self.logger.info("########## Training Finished ###########")
        if dist.get_rank() == 0:
            top1s, cls_recalls, cls_precs = self.validate(bias_validate=False)
            output = "Eval Results: \n"
            if top1s[0] > best_top1:
                best_top1 = top1s[0]
                best_epoch = self.epoch
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

            # self.logger.info("bias accs are")
            # print()
            # for line in bias_accs:
            #     print(','.join(list(map(str, line.tolist()))))

    def update_loss_weights(self):
        n_class = self.cfg.model.head.n_classes[0]

        losses = []
        labels = []

        self.model.train()
        mean_loss = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                output, loss_dict = self.model(inputs, target)

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
                if len(class_indices) == 0:
                    mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
                else:
                    class_loss = losses[class_indices]
                    mean_loss.append(class_loss.mean())
                class_nums.append(len(class_indices))

            cls_loss_weight = torch.tensor(mean_loss, requires_grad=False).cuda()
            cls_loss_weight = cls_loss_weight - cls_loss_weight.min()

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weights_probs(self):
        n_class = self.cfg.model.head.n_classes[0]

        probs = []
        labels = []

        self.model.train()
        mean_prob = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                output, loss_dict = self.model(inputs, target)

                prob = torch.exp(-loss_dict['cls_1_loss'])
                prob = prob + prob**2 / 2.
                label = target[:, 0]

                probs.append(prob)
                labels.append(label)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()

            # probs_values = torch.exp(-probs)
            # probs_mu = torch.mean(probs_values).cuda()
            # probs_std = torch.std(probs_values).cuda()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                #     mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
                # else:
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())

                class_nums.append(len(class_indices))
            # print(mean_prob)


            cls_loss_weight = -torch.tensor(mean_prob, requires_grad=False).cuda()
            # cls_loss_weight[0] *= 10
            cls_loss_weight = torch.exp(cls_loss_weight)

            cls_loss_weight = cls_loss_weight - cls_loss_weight.min() + 1e-4

            # cls_loss_weight = torch.exp(-(cls_loss_weight - probs_mu.cuda())**2/(2*probs_std**2))



            # cls_loss_weight[0] /= 10.
            # cls_loss_weight = 1./ cls_loss_weight
            # cls_loss_weight = cls_loss_weight - cls_loss_weight.min()

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weight_v1(self):
        n_class = self.cfg.model.head.n_classes[0]

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
                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target[:, 0].contiguous(), num_classes=n_class).float().cuda()), dim=1)
                prob = prob
                label = target[:, 0]

                probs.append(prob)
                labels.append(label)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()

            # probs_values = torch.exp(-probs)
            # probs_mu = torch.mean(probs_values).cuda()
            # probs_std = torch.std(probs_values).cuda()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                #     mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
                # else:
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())

                class_nums.append(len(class_indices))
            # print(mean_prob)

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * 0.9999

            sample_ratio = torch.tensor(self.cfg.trainer.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_error_rate**2 * (1-cls_error_rate))) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            # cls_loss_weight[0] *= 10
            # cls_loss_weight = torch.exp(cls_loss_weight)

            # cls_loss_weight = cls_loss_weight - cls_loss_weight.min() + 1e-4

            # cls_loss_weight = torch.exp(-(cls_loss_weight - probs_mu.cuda())**2/(2*probs_std**2))

            # cls_loss_weight[0] /= 10.
            # cls_loss_weight = 1./ cls_loss_weight
            # cls_loss_weight = cls_loss_weight - cls_loss_weight.min()

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weight_v2(self):
        n_class = self.cfg.model.head.n_classes[0]

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

                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target[:, 0].contiguous(), num_classes=n_class).float().cuda()), dim=1)
                var = prob * (1-prob)
                label = target[:, 0]

                probs.append(prob)
                labels.append(label)
                vars.append(var)
            probs = torch.cat(probs, dim=0)
            labels = torch.cat(labels, dim=0)
            vars = torch.cat(vars, dim=0)
            probs = sync_tensor_across_gpus(probs).cpu()
            labels = sync_tensor_across_gpus(labels).cpu()
            vars = sync_tensor_across_gpus(vars).cpu()

            # probs_values = torch.exp(-probs)
            # probs_mu = torch.mean(probs_values).cuda()
            # probs_std = torch.std(probs_values).cuda()

            class_nums = []
            for i in range(n_class):
                class_indices = torch.nonzero(labels == i)
                assert len(class_indices) > 0
                #     mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
                # else:
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())
                class_var = vars[class_indices]
                mean_var.append(class_var.mean())

                class_nums.append(len(class_indices))
            # print(mean_prob)

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            cls_var_rate = torch.tensor(mean_var, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * 0.9999

            sample_ratio = torch.tensor(self.cfg.trainer.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_error_rate * cls_var_rate)) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            # cls_loss_weight[0] *= 10
            # cls_loss_weight = torch.exp(cls_loss_weight)

            # cls_loss_weight = cls_loss_weight - cls_loss_weight.min() + 1e-4

            # cls_loss_weight = torch.exp(-(cls_loss_weight - probs_mu.cuda())**2/(2*probs_std**2))

            # cls_loss_weight[0] /= 10.
            # cls_loss_weight = 1./ cls_loss_weight
            # cls_loss_weight = cls_loss_weight - cls_loss_weight.min()

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            self.cls_loss_weight = cls_loss_weight

    def update_loss_weight_v3(self):
        n_class = self.cfg.model.head.n_classes[0]

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

                prob = torch.sum(torch.softmax(output, 1) * (F.one_hot(target[:, 0].contiguous(), num_classes=n_class).float().cuda()), dim=1)
                var = prob * (1-prob)
                embeddings_norms = (1-prob) * embeddings_norms

                label = target[:, 0]

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
                #     mean_loss.append(-np.log(0.05) / self.cls_loss_weight[i])
                # else:
                class_prob = probs[class_indices]
                mean_prob.append(class_prob.mean())
                class_var = vars[class_indices]
                mean_var.append(class_var.mean())
                mean_norm.append(embeddings_norms[class_indices].mean())

                class_nums.append(len(class_indices))

            cls_error_rate = 1 - torch.tensor(mean_prob, requires_grad=False).cuda()
            cls_var_rate = torch.tensor(mean_var, requires_grad=False).cuda()
            cls_embeddings_rate = torch.tensor(mean_norm, requires_grad=False).cuda()
            expected_error_rate = cls_error_rate.min() * 0.9999

            sample_ratio = torch.tensor(self.cfg.trainer.sample_ratio, requires_grad=False).cuda() ** 1.0
            cls_loss_weight = ((cls_error_rate - expected_error_rate) / (cls_embeddings_rate * cls_var_rate)) / sample_ratio

            cls_error_rate_str = ', '.join([f"{i:0.4f}" for i in cls_error_rate.cpu().numpy()])
            self.logger.info(f" error_rate: {cls_error_rate_str}")

            cls_loss_weight = cls_loss_weight / cls_loss_weight.sum() * n_class + 1e-6
            self.cls_loss_weight = cls_loss_weight

    def train(self):
        # per epoch loss weight update

        if self.cfg.model.head.loss =='ib-ce':
            self.model.module.head.loss = 'ce'

        if self.epoch >= 10:
            self.model.module.head.loss = self.cfg.model.head.loss

        if self.epoch >= 12:
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

        n_classifier = len(self.cfg.data.train.n_classes)
        n_class = self.cfg.model.head.n_classes[0]

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        labeled_losses = [AverageMeter('Loss', ':.4e') for i in range(n_classifier)]

        self.model.train()

        end = time.time()
        for i, (inputs, target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            target = target.cuda()
            output, loss_dict = self.model(inputs, target, self.cls_loss_weight, flag=True, epoch=self.epoch)

            loss = sum([t.mean() for t in loss_dict.values()])

            acc1 = accuracy(output[0], target[:, 0], topk=(1, ))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))

            for j in range(n_classifier):
                labeled_losses[j].update(loss_dict[f'cls_{j+1}_loss'].mean().item(), inputs.size(0))


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
                for j in range(n_classifier):
                    output += f"cls_{j+1}/loss_{j+1} = {labeled_losses[j].val:.4f}\t"
                self.logger.info(output)

        if dist.get_rank() == 0:
            self.writer.add_scalar('loss/train_all', losses.avg, self.epoch)
            self.writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], self.epoch)
            for i in range(n_classifier):
                self.writer.add_scalar(f'loss/train_cls_{i+1}', labeled_losses[i].avg, self.epoch)

    def validate(self, record=True, bias_validate=False, save_probs=False, save_embeddings=False):
        batch_time = AverageMeter('Time', ':6.3f')
        n_classes = self.cfg.data.train.n_classes
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
            for i, (inputs, target) in enumerate(self.val_loader):
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

                    # if j==0:
                    #     output[0] = output[0][:, [0,1,3,4,5,6]]
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
                    cf = confusion_matrix(all_targets[i], all_preds[i], labels=range(n_classes[i])).astype(float)
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

            if record and dist.get_rank()==0:
                for i in range(n_classifier):
                    self.writer.add_scalar(f'Eval/acc_top1_{i + 1}', top1s[i].avg, self.epoch)
                    self.writer.add_scalars(f'Eval/acc_cls_{i + 1}', {str(i): x for i, x in enumerate(cls_accs[i])},
                                            self.epoch)

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


            if bias_validate:
                n_classes_sum = sum(n_classes)
                bias_accs = np.zeros((n_classes_sum, n_classes_sum), dtype=np.float32)

                row_index = 0
                for i in range(n_classifier):
                    n_class = n_classes[i]
                    for j in range(n_class):
                        target_mask = np.array(all_targets[i]) == j
                        col_index = 0
                        for k in range(n_classifier):
                            if i == k:
                                col_index += n_classes[k]
                                continue
                            all_target = (np.array(all_targets[k])[target_mask]).tolist()
                            all_pred =(np.array(all_preds[k])[target_mask]).tolist()
                            cf = confusion_matrix(all_target, all_pred, labels=range(n_classes[k])).astype(float)
                            cls_cnt = cf.sum(axis=1)
                            cls_hit = np.diag(cf)
                            cls_acc = cls_hit / cls_cnt
                            bias_accs[row_index, col_index:col_index + n_classes[k]] = cls_acc
                            col_index += n_classes[k]
                        row_index += 1
                return [top1.avg for top1 in top1s], cls_accs, bias_accs

            return [top1.avg for top1 in top1s], cls_accs, cls_precs

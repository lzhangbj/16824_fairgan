# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings
import shutil
import time
from datetime import timedelta
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from balface.models import build_model
from balface.datasets import build_dataset
from balface.utils import get_root_logger, AverageMeter, accuracy
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

class SSPTrainer:
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
            self.resume(cfg.resume_from)
        self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, self.optimizer, last_epoch=self.epoch)

        train_dataset = build_dataset(cfg.data.train)
        # prepare data loaders
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.train.samples_per_gpu,
            num_workers=cfg.data.train.workers_per_gpu,
            sampler=train_sampler
        )

        # no validation


    def resume(self, resume_from):
        save_content = torch.load(resume_from)
        epoch = save_content['epoch']
        statedict = save_content['state_dict']
        opt_state = save_content['opt_state']

        self.model.module.load_state_dict(statedict)
        self.optimizer.load_state_dict(opt_state)
        self.epoch = epoch

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
        max_epochs = self.cfg.trainer.max_epochs

        init_epoch = self.epoch + 1

        train_time = AverageMeter('Time', ':6.3f')

        end = time.time()
        for self.epoch in range(init_epoch, max_epochs+1):
            self.lr_scheduler.step()
            self.train()

            if self.epoch % self.cfg.trainer.save_freq == 0:
                self.save_checkpoint(latest=True)

            train_time.update(time.time() - end)
            end = time.time()

            eta = train_time.avg * (max_epochs - self.epoch)
            format_eta = str(timedelta(seconds=eta))
            self.logger.info(f'######### ETA {format_eta} ##########')

        self.logger.info("########## Training Finished ###########")

    def train(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.train()

        end = time.time()
        for i, (inputs1, inputs2) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()
            output, loss_dict = self.model(inputs1, inputs2)

            loss = sum([t.mean() for t in loss_dict.values()])

            losses.update(loss.item(), inputs1.size(0))

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 5)

            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.cfg.trainer.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    self.epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=self.optimizer.param_groups[-1]['lr']))
                self.logger.info(output)
        self.logger.info(f"Epoch [{self.epoch}] training loss {losses.avg:.4f}")

        if dist.get_rank() == 0:
            self.writer.add_scalar('loss/train', losses.avg, self.epoch)
            self.writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
            self.writer.add_scalar('acc/train_top5', top5.avg, self.epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], self.epoch)

        return losses.avg

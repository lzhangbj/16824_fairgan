import argparse
import logging
import os
import os.path as osp
import time
import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.distributed as dist
import torch.utils.data.distributed

from mmcv import Config, DictAction

from balface.models import build_model
from balface.datasets import build_dataset
from balface.utils import AverageMeter, get_root_logger,set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', ],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    os.makedirs(cfg.work_dir, exist_ok=True)
    if args.resume_from:
        cfg.resume_from = args.resume_from

    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    print("$$$$$$$$$$$$$$ word size is ", world_size)
    rank = dist.get_rank()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    cfg.local_rank = local_rank

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)

    logger.info(f'Config:\n{cfg.pretty_text}')
    seed = args.seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.logger = logger

    if cfg.trainer.name == 'ssp':
        from balface.apis import SSPTrainer as Trainer
    elif cfg.trainer.name == 'ssl':
        from balface.apis import SSLTrainer as Trainer
    elif cfg.trainer.name == 'cls':
        from balface.apis import CLSTrainer as Trainer
    elif cfg.trainer.name == 'denoise_cls':
        from balface.apis import DenoiseCLSTrainer as Trainer
    elif cfg.trainer.name == 'fixmatch':
        from balface.apis import FixMatchTrainer as Trainer
    elif cfg.trainer.name == 'cluster':
        from balface.apis import ClusterTrainer as Trainer
    elif cfg.trainer.name == 'sslssp':
        from balface.apis import SSLSSPTrainer as Trainer
    elif cfg.trainer.name == 'rsc':
        from balface.apis import RSCTrainer as Trainer
    elif cfg.trainer.name == 'cutmix':
        from balface.apis import CutmixTrainer as Trainer
    elif cfg.trainer.name == 'cifar':
        from balface.apis import CifarTrainer as Trainer
    elif cfg.trainer.name == 'feataug':
        from balface.apis import FeataugTrainer as Trainer
    elif cfg.trainer.name == 'latentaug':
        from balface.apis import LatentaugTrainer as Trainer
    else:
        raise Exception()

    trainer = Trainer(cfg)
    trainer.run()

    exit(0)

if __name__ == '__main__':
    main()







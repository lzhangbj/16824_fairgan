import logging
import os
import sys

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='balface', log_file=log_file, log_level=log_level)
    logger.propagate = False

    return logger


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

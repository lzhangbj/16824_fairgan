import torch.distributed as dist
from .fc_fuser import FCConcatFuser, FCSumFuser

def build_feat_fuser(feat_fuser_cfg):
    feat_fuser_name = feat_fuser_cfg.name
    if feat_fuser_name == 'fc_concat_fuser':
        model = FCConcatFuser(**feat_fuser_cfg)
    elif feat_fuser_name == 'fc_sum_fuser':
        model = FCSumFuser(**feat_fuser_cfg)
    else:
        raise Exception()
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f" feat fuser parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

__all__ = [
    'build_feat_fuser'
]
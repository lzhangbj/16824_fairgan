import torch.distributed as dist

from .resnet import resnet18, resnet34, torch_resnet34, head_torch_resnet34, rsc_torch_resnet34
from .efficientnet import efficientnet
from .swin_transformer import swintransformer
from .w_encoder import WEncoder
from .resnet_cifar import resnet32
from .e4e import Encoder4Editing

def build_backbone(backbone_cfg):
    backbone_name = backbone_cfg.name
    if backbone_name == 'resnet34':
        model = resnet34()
    elif backbone_name == 'resnet32':
        model = resnet32()
    elif backbone_name == 'torch_resnet34':
        model = torch_resnet34()
    elif backbone_name == 'rsc_torch_resnet34':
        model = rsc_torch_resnet34()
    elif backbone_name == 'head_torch_resnet34':
        model = head_torch_resnet34()
    elif backbone_name == 'torch_efficientnet':
        model = efficientnet()
    elif backbone_name == "torch_swintransformer":
        model = swintransformer()
    elif backbone_name == "pretrained_w_encoder":
        model = WEncoder()
    elif backbone_name == "pretrained_e4e":
        model = Encoder4Editing()
    else:
        raise Exception()

    if dist.is_initialized() and dist.get_rank() == 0:
        print(f" backbone parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

__all__ = [
    'build_backbone'
]




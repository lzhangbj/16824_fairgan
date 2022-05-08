import torch.distributed as dist

from .fc_head import ContrastiveHead, ClassifierHead, MultiClassifierHead, MultiClassifierContrastiveHead

def build_head(head_cfg):
    input_dim = head_cfg.input_dim
    head_name = head_cfg.name
    loss = head_cfg.loss

    if head_name == 'contrastive_head':
        hidden_dims = head_cfg.hidden_dims
        embed_dim = head_cfg.embed_dim
        T = head_cfg.T
        model = ContrastiveHead(embed_dim, input_dim, hidden_dims, T)
    elif head_name == 'classifier_head':
        hidden_dims = head_cfg.hidden_dims
        n_class = head_cfg.n_class
        norm_weights = head_cfg.norm_weights
        model = ClassifierHead(n_class, input_dim, hidden_dims, loss=loss, norm_weights=norm_weights)
    elif head_name == 'multiclassifier_head':
        hidden_dims = head_cfg.hidden_dims
        n_classes = head_cfg.n_classes
        norm_weights = head_cfg.norm_weights
        model = MultiClassifierHead(n_classes, input_dim, hidden_dims, loss=loss, norm_weights=norm_weights)
    elif head_name == 'multiclassifier_contrastive_head':
        cls_hidden_dim = head_cfg.cls_hidden_dim
        ssp_hidden_dim = head_cfg.ssp_hidden_dim
        embed_dim = head_cfg.ssp_embed_dim
        n_classes = head_cfg.n_classes
        model = MultiClassifierContrastiveHead(n_classes, input_dim,
                                              cls_hidden_dim, embed_dim, ssp_hidden_dim)
    else:
        raise Exception()

    if dist.is_initialized() and dist.get_rank() == 0:
        print(f" head parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model

__all__ = [
    'build_head'
]


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from balface.models.backbone import build_backbone
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus


class SSLSSPRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(SSLSSPRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)
        n_class = model_cfg.head.n_classes[0]
        self.n_class = n_class

        self.ema_backbone = build_backbone(model_cfg.backbone)
        self.ema_head = build_head(model_cfg.head)
        self.ema_momentum = model_cfg.ema_momentum
        for param in self.ema_backbone.parameters():
            param.detach_()
            param.requires_grad = False
        for param in self.ema_head.parameters():
            param.detach_()
            param.requires_grad = False

        self.loss = model_cfg.loss

    @torch.no_grad()
    def update_ema_module(self, model, shadow):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(model.named_parameters())
        shadow_params = OrderedDict(shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.ema_momentum) * (shadow_params[name] - param))

        model_buffers = OrderedDict(model.named_buffers())
        shadow_buffers = OrderedDict(shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def update_ema(self):
        self.update_ema_module(self.backbone, self.ema_backbone)
        self.update_ema_module(self.head, self.ema_head)

    def get_embeddings(self, inputs):
        return self.ema_backbone(inputs)

    def init_ema(self):
        self.ema_backbone.load_state_dict(self.backbone.state_dict())
        self.ema_head.load_state_dict(self.head.state_dict())
        for param in self.ema_backbone.parameters():
            param.requires_grad = False
        for param in self.ema_head.parameters():
            param.requires_grad = False

    def compute_pseudolabel(self, logits, thresh=0.95):
        if self.loss == 'ce':
            pseudo_labels = torch.stack([torch.argmax(logit, dim=1) for logit in logits], dim=1)
            probs = [torch.max(torch.softmax(logit, dim=1), dim=1)[0] for logit in logits]
            mask = torch.stack([prob >= thresh for prob in probs], dim=1).float()
        elif self.loss == 'kld':
            pseudo_labels = torch.softmax(logits[0], dim=1)  # (B, 7)
            max_probs = torch.max(pseudo_labels, dim=1)[0]
            mask = (max_probs >= thresh).float()

            pseudo_labels = pseudo_labels.unsqueeze(1)
            mask = mask.unsqueeze(1)
        return pseudo_labels, mask

    def forward(self, label_input1, label_input2=None,
                weak_input=None, strong_input1=None, strong_input2=None,
                labels=None, threshold=0.95,
                label_weights=None, unlabel_weights=None):

        if self.training:
            B_labeled = label_input1.size(0)
            B_unlabeled = weak_input.size(0)
            input = torch.cat([label_input1, label_input2, strong_input1, strong_input2])
            embeddings = self.backbone(input)

            labeled_embeddings1 = embeddings[:B_labeled]
            labeled_embeddings2 = embeddings[B_labeled:2*B_labeled]
            strong_embeddings1 = embeddings[-2*B_unlabeled:-B_unlabeled]
            strong_embeddings2 = embeddings[-B_unlabeled:]

            self.eval()
            with torch.no_grad():
                weak_embeddings = self.ema_backbone(weak_input)
                logits = self.ema_head.compute_cls_logits(weak_embeddings)
            pseudo_labels, masks = self.compute_pseudolabel(logits, threshold)

            self.train()

            # compute pseudo_labels adaptively if it is not provided
            if self.loss == 'ce':
                features_list, loss_dict = self.head(labeled_embeddings1, labeled_embeddings2, strong_embeddings1, strong_embeddings2,
                                                     labels, pseudo_labels, masks,
                                                     label_weights=label_weights, unlabel_weights=unlabel_weights)
            elif self.loss == 'kld':
                features_list, loss_dict = self.head(labeled_embeddings1, labeled_embeddings2, strong_embeddings1, strong_embeddings2,
                                                     labels, pseudo_labels, masks,
                                                     label_weights=label_weights, unlabel_weights=unlabel_weights)
            else:
                raise Exception()

            return features_list, loss_dict, pseudo_labels

        else:
            embeddings = self.ema_backbone(label_input1)
            logits = self.ema_head.compute_cls_logits(embeddings)
            probs = [F.softmax(logit, dim=1) for logit in logits]
            return probs, {}

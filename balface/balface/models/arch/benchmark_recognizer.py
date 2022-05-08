import numpy as np
import torch
import torch.nn as nn

from balface.models.backbone import build_backbone
from balface.models.head import build_head

class BenchmarkRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(BenchmarkRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)
        self.use_weight = model_cfg.use_weight

    def forward(self, inputs, labels=None, loss_weight=None, one_hot_labels=False):
        embeddings = self.backbone(inputs)

        label_weights = None
        # if self.use_weight:
        #     label_weights = torch.tensor([1., 10, 10, 10, 10, 10, 10.]).float().cuda()

        features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight, one_hot_labels=one_hot_labels)

        return features_list, loss_dict

    def get_embeddings(self, inputs):
        return self.backbone(inputs)




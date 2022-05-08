import numpy as np
import torch
import torch.nn as nn

from balface.models.backbone import build_backbone
from balface.models.head import build_head

class SSPRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(SSPRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)

    def forward(self, input1, input2):
        B = input1.size(0)
        inputs = torch.cat([input1, input2])
        embeddings = self.backbone(inputs)

        embeddings1 = embeddings[:B]
        embeddings2 = embeddings[B:]

        features_list, loss_dict = self.head(embeddings1, embeddings2)

        return features_list, loss_dict





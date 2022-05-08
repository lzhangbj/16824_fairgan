import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from efficientnet_pytorch import EfficientNet

class TorchEfficientNet(nn.Module):
    def __init__(self):
        super(TorchEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', include_top=False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(0.5)

    # @torch.no_grad()
    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.squeeze(2).squeeze(2)
        feat = self.dropout(feat)
        return feat

def efficientnet():
    # B0: 4,007,548
    # B3: 10,696,232
    return TorchEfficientNet()

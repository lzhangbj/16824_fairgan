import torch
import torch.nn as nn
import torch.nn.functional as F

class FCConcatFuser(nn.Module):
    def __init__(self, in_feat1, in_feat2, out_feat, hidden_dims=(), *args, **kwargs):
        super(FCConcatFuser, self).__init__()
        self.in_feat = in_feat1 + in_feat2
        self.out_feat = out_feat
        self.fc = nn.ModuleList()
        in_feat = self.in_feat
        for i,dim in enumerate(hidden_dims):
            self.fc.append(nn.Linear(in_feat, dim))
            self.fc.append(nn.ReLU(inplace=True))
            in_feat = dim
        self.fc.append(nn.Linear(in_feat, out_feat))
    
    def forward(self, cls_feat, latent_feat):
        feat = torch.cat([cls_feat, latent_feat], dim=1)
        for mod in self.fc:
            feat = mod(feat)
        return feat
    


class FCSumFuser(nn.Module):
    def __init__(self, in_feat1, in_feat2, hidden_dims=(), *args, **kwargs):
        super(FCSumFuser, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.fc = nn.ModuleList()
        in_feat = self.in_feat2
        for i, dim in enumerate(hidden_dims):
            self.fc.append(nn.Linear(in_feat, dim))
            self.fc.append(nn.ReLU(inplace=True))
            in_feat = dim
        self.fc.append(nn.Linear(in_feat, in_feat1))

    def forward(self, cls_feat, latent_feat):
        for mod in self.fc:
            latent_feat = mod(latent_feat)
        return latent_feat + cls_feat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.models.backbone import build_backbone
from balface.models.head import build_head

class SSLRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(SSLRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)

    def compute_pseudolabel(self, embeddings):
        logits = self.head.compute_cls_logits(embeddings)
        pseudo_labels = torch.stack([torch.argmax(logit, dim=1) for logit in logits], dim=1)
        return pseudo_labels

    def get_embeddings(self, inputs):
        return self.backbone(inputs)

    def forward(self, label_input, unlabel_input=None, labels=None, unlabeled_labels=None):
        if self.training:
            B_labeled = label_input.size(0)
            B_unlabeled = unlabel_input.size(0)
            input = torch.cat([label_input, unlabel_input])

            embeddings = self.backbone(input)

            labeled_embeddings = embeddings[:B_labeled]
            unlabeled_embeddings = embeddings[B_labeled:]

            pseudo_labels = unlabeled_labels
            # compute pseudo_labels adaptively if it is not provided
            if self.training and pseudo_labels is None:
                self.eval()
                with torch.no_grad():
                    pseudo_labels = self.compute_pseudolabel(unlabeled_embeddings)
                self.train()
            features_list, loss_dict = self.head(labeled_embeddings, unlabeled_embeddings, labels, pseudo_labels)

            return features_list, loss_dict
        else:
            embeddings = self.backbone(label_input)
            logits = self.head.compute_cls_logits(embeddings)
            probs = [F.softmax(logit, dim=1) for logit in logits]
            return probs, {}







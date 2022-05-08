import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from balface.models.backbone import build_backbone
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus


class ClusterRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(ClusterRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)
        self.n_class = model_cfg.head.n_classes[0]
        self.feat_dim = 512

        centers = torch.randn(self.n_class, self.feat_dim)
        centers = F.normalize(centers, dim=1)
        self.clusters = nn.Parameter(centers, requires_grad=True)
        self.loss = model_cfg.loss
        self.use_pseudo_label = model_cfg.use_pseudo_label
        self.use_head = model_cfg.use_head
        if not self.use_head:
            for param in self.head.parameters():
                param.requires_grad = False
        self.use_weight = model_cfg.use_weight
        if self.use_weight:
            weight = torch.ones(7).float().cuda()/7.
            self.weights = nn.Parameter(weight, requires_grad=False)
            self.weight_momentum = 0.9

    def get_embeddings(self, inputs):
        return self.backbone(inputs)

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
        else:
            raise Exception()

        return pseudo_labels, mask

    def compute_cluster_labels(self, embeddings):
        # embeddings = F.normalize(embeddings, dim=1)
        # cluster_embeddings = F.normalize(self.clusters, dim=1)
        # distances = torch.cdist(embeddings.unsqueeze(0), cluster_embeddings.unsqueeze(0)).squeeze(0)
        logit = self.head.compute_cls_logits(embeddings)[0]
        probs = torch.argmax(logit, dim=1)

        return probs

    def compute_cluster_loss(self, embeddings, labels, prefix="", weight=None):
        embeddings = F.normalize(embeddings, dim=1)
        cluster_embeddings = F.normalize(self.clusters, dim=1)
        distances = torch.cdist(embeddings.unsqueeze(0), cluster_embeddings.unsqueeze(0)).squeeze(0) # (B, C)

        one_hot_labels = torch.eye(self.n_class).cuda()[labels].bool()

        B = embeddings.size(0)
        other_distances = distances[~one_hot_labels].contiguous().view(B, self.n_class-1)

        second_distances = torch.min(other_distances, dim=1)[0]

        center_distances = torch.gather(distances, 1, labels.unsqueeze(1))

        margin_loss = F.relu(center_distances - second_distances + 0.8).mean()

        return {prefix+"center_loss": margin_loss}

    def cluster_loss(self):
        reg_loss = torch.mean(torch.abs(torch.norm(self.clusters, dim=1)-1))

        cluster_embeddings = F.normalize(self.clusters, dim=1)
        pairwise_dists = torch.cdist(cluster_embeddings.unsqueeze(0), cluster_embeddings.unsqueeze(0)).squeeze(0) # (c, c)

        one_hot_labels = torch.eye(self.n_class).cuda().bool()
        other_distances = pairwise_dists[~one_hot_labels].view(self.n_class, self.n_class - 1)
        mean_dist = other_distances.mean()
        std_loss = torch.abs(other_distances-mean_dist).mean()

        pairwise_dists = pairwise_dists + (torch.eye(self.n_class) * 100).cuda()
        margin_loss = (2-torch.min(pairwise_dists, dim=1)[0]).mean()

        return {
            'cluster_reg_loss': reg_loss,
            'cluster_margin_loss': margin_loss,
            'cluster_std_loss': std_loss
        }

    def forward(self, label_input, weak_input=None, strong_input=None, labels=None, threshold=0.95):
        if self.training:
            B_labeled = label_input.size(0)
            B_unlabeled = weak_input.size(0)
            input = torch.cat([label_input, weak_input, strong_input])
            embeddings = self.backbone(input)

            labeled_embeddings = embeddings[:B_labeled]
            weak_embeddings = embeddings[B_labeled:-B_unlabeled]
            strong_embeddings = embeddings[-B_unlabeled:]

            with torch.no_grad():
                self.eval()
                unlabeled_cluster_labels = self.compute_cluster_labels(weak_embeddings)
                self.train()
            label_weights = None
            unlabel_weights = None
            if self.use_weight:
                squeezed_pseudo_labels = sync_tensor_across_gpus(unlabeled_cluster_labels)
                one_hot_pseudo_labels = torch.eye(7).float().cuda()[squeezed_pseudo_labels] # (B, 7)
                updated_weights = 1. / (one_hot_pseudo_labels.sum(dim=0) / one_hot_pseudo_labels.sum() + 1e-6)
                updated_weights = updated_weights / updated_weights.sum()
                self.weights.data = self.weight_momentum*self.weights.data + updated_weights*(1-self.weight_momentum)
                self.weights.data = self.weights.data / self.weights.data.sum()
                unlabel_weights = self.weights.data
                label_weights = torch.tensor([1., 10, 10, 10, 10, 10, 10.]).float().cuda()
                label_weights = label_weights / label_weights.sum()

            labeled_cluster_loss_dict = self.compute_cluster_loss(labeled_embeddings, labels[:, 0], 'labeled_', weight=label_weights)
            unlabeled_weak_cluster_loss_dict = self.compute_cluster_loss(weak_embeddings, unlabeled_cluster_labels, 'weak_', weight=unlabel_weights)
            unlabeled_strong_cluster_loss_dict = self.compute_cluster_loss(strong_embeddings, unlabeled_cluster_labels, 'strong_', weight=unlabel_weights)
            cluster_loss = self.cluster_loss()

            masks = None
            pseudo_labels = unlabeled_cluster_labels.unsqueeze(1)

            cluster_loss_dict = {}
            cluster_loss_dict.update(labeled_cluster_loss_dict)
            cluster_loss_dict.update(unlabeled_weak_cluster_loss_dict)
            cluster_loss_dict.update(unlabeled_strong_cluster_loss_dict)
            cluster_loss_dict.update(cluster_loss)

            # compute pseudo_labels adaptively if it is not provided
            if not self.use_pseudo_label:
                strong_embeddings = None
                pseudo_labels = None
                masks = None

            if self.loss == 'ce':
                features_list, loss_dict = self.head(labeled_embeddings, strong_embeddings, labels, pseudo_labels, masks,
                                                     label_weights=label_weights, unlabel_weights=unlabel_weights)
            elif self.loss == 'kld':
                features_list, loss_dict = self.head(labeled_embeddings, strong_embeddings, labels, pseudo_labels, masks)
            else:
                raise Exception()

            loss_dict.update(cluster_loss_dict)

            return features_list, loss_dict
        else:
            embeddings = self.backbone(label_input)

            if not self.use_head:
                preds = self.compute_cluster_labels(embeddings)
                probs = [torch.eye(self.n_class).cuda()[preds],]

                return probs, {}

            logits = self.head.compute_cls_logits(embeddings)
            probs = [F.softmax(logit, dim=1) for logit in logits]
            return probs, {}

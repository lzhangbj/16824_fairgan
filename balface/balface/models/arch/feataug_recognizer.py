import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.models.backbone import build_backbone
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus

class FeataugRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(FeataugRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)
        self.use_weight = model_cfg.use_weight
        self.feat_aug = model_cfg.feat_aug
        self.n_class = model_cfg.head.n_classes[0]
        self.aug_num_per_gpu = model_cfg.aug_num_per_gpu
        aug_ratio = np.array(model_cfg.aug_ratio)
        self.aug_ratio = aug_ratio / aug_ratio.sum()
        self.blend_alpha = model_cfg.blend_alpha
        self.feat_norm = model_cfg.feat_norm
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, inputs, labels=None, loss_weight=None, one_hot_labels=False):
        embeddings = self.backbone(inputs)
        h, w = embeddings.size()[2:]

        assert not one_hot_labels

        if self.feat_aug == 'none' or not self.training:
            embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))

            if self.feat_norm:
                embeddings = F.normalize(embeddings, dim=1)
            features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight, one_hot_labels=one_hot_labels)

        elif self.feat_aug == 'inst-dom':
            embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
            global_embeddings = sync_tensor_across_gpus(embeddings)
            global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
            B = global_targets.size(0)

            cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

            sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
            sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
            sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
            sampled_embeddings1 = global_embeddings[sampled1_indices]
            sampled_targets1 = global_targets[sampled1_indices]

            sampled2_indices = np.random.randint(0, B, self.aug_num_per_gpu)
            sampled_embeddings2 = global_embeddings[sampled2_indices]
            # sampled_targets2 = global_targets[sampled2_indices]

            blend_alphas = torch.tensor(np.random.uniform(0., self.blend_alpha, size=self.aug_num_per_gpu)).float().cuda().unsqueeze(1)
            aug_embeddings = sampled_embeddings2 * blend_alphas + (1-blend_alphas) * sampled_embeddings1

            aug_targets = sampled_targets1

            embeddings = torch.cat([embeddings, aug_embeddings])
            labels = torch.cat([labels[:, 0], aug_targets]).unsqueeze(1)

            if self.feat_norm:
                embeddings = F.normalize(embeddings, dim=1)
            features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
                                                 one_hot_labels=False)
            features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]

        elif self.feat_aug == 'inst-convex':
            embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
            global_embeddings = sync_tensor_across_gpus(embeddings)
            global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
            B = global_targets.size(0)
            cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

            sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
            sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
            sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
            sampled_embeddings1 = global_embeddings[sampled1_indices]
            sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class)

            sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
            sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
            sampled2_indices = np.concatenate(
                [np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
            sampled_embeddings2 = global_embeddings[sampled2_indices]
            sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)

            blend_alphas = torch.tensor(np.random.uniform(0., self.blend_alpha, size=self.aug_num_per_gpu)).float().cuda().unsqueeze(1)

            aug_embeddings = sampled_embeddings2 * blend_alphas + (1-blend_alphas) * sampled_embeddings1

            aug_targets = sampled_targets1 * (1-blend_alphas) + sampled_targets2 * blend_alphas

            embeddings = torch.cat([embeddings, aug_embeddings])
            labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

            if self.feat_norm:
                embeddings = F.normalize(embeddings, dim=1)
            features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
                                                 one_hot_labels=True)
            features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]

        elif self.feat_aug == 'region-convex':
            # embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
            global_embeddings = sync_tensor_across_gpus(embeddings)
            global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
            B = global_targets.size(0)
            cls_indices_list = [torch.nonzero(global_targets == i).squeeze(1).cpu().numpy() for i in
                                range(self.n_class)]

            sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True,
                                            p=self.aug_ratio)
            sampled_cls1_cls_num_list = [np.sum(sampled_cls1 == i, dtype=np.int32) for i in range(self.n_class)]
            sampled1_indices = np.concatenate(
                [np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in
                 range(self.n_class)])
            sampled_embeddings1 = global_embeddings[sampled1_indices]
            sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class)

            sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True,
                                            p=self.aug_ratio)
            sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
            sampled2_indices = np.concatenate(
                [np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in
                 range(self.n_class)])
            sampled_embeddings2 = global_embeddings[sampled2_indices]
            sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)

            blend_lateral = np.random.randint(1, min(h, w), size=self.aug_num_per_gpu)
            blend_x0 = np.random.randint(0, w-blend_lateral+1)
            blend_y0 = np.random.randint(0, h-blend_lateral+1)

            blend_mask = torch.zeros((self.aug_num_per_gpu, 1, h, w)).float()
            for i in range(self.aug_num_per_gpu):
                x0 = blend_x0[i]
                y0 = blend_y0[i]
                x1 = x0+blend_lateral[i]
                y1 = y0+blend_lateral[i]
                blend_mask[i, :, y0:y1, x0:x1] = 1
            blend_mask = blend_mask.contiguous().cuda()

            blend_alphas = torch.tensor(blend_lateral * blend_lateral / 49.).float().cuda().unsqueeze(1)

            aug_embeddings = sampled_embeddings2 * blend_mask + (1 - blend_mask) * sampled_embeddings1

            aug_targets = sampled_targets1 * (1 - blend_alphas) + sampled_targets2 * blend_alphas

            embeddings = torch.cat([embeddings, aug_embeddings])
            embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))

            labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

            if self.feat_norm:
                embeddings = F.normalize(embeddings, dim=1)
            features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
                                                 one_hot_labels=True)
            features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]

        elif self.feat_aug == 'channel-convex':
            embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
            global_embeddings = sync_tensor_across_gpus(embeddings)
            global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
            B = global_targets.size(0)
            cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

            sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
            sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
            sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
            sampled_embeddings1 = global_embeddings[sampled1_indices]
            sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class)

            sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
            sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
            sampled2_indices = np.concatenate(
                [np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
            sampled_embeddings2 = global_embeddings[sampled2_indices]
            sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)


            blend_alphas = np.random.randint(1, 512, size=self.aug_num_per_gpu)

            blend_mask = torch.zeros((self.aug_num_per_gpu, 512)).float()
            for i in range(self.aug_num_per_gpu):
                indices = np.random.choice(512, blend_alphas[i], replace=False)
                blend_mask[i, indices] = 1.0
            blend_mask = blend_mask.contiguous().cuda()
            blend_alphas = blend_alphas.astype(np.float32) / 512.
            blend_alphas = torch.tensor(blend_alphas).float().cuda().unsqueeze(1)

            aug_embeddings = sampled_embeddings2 * blend_mask + (1-blend_mask) * sampled_embeddings1
            aug_targets = sampled_targets1 * (1-blend_alphas) + sampled_targets2 * blend_alphas

            embeddings = torch.cat([embeddings, aug_embeddings])
            labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

            if self.feat_norm:
                embeddings = F.normalize(embeddings, dim=1)
            features_list, loss_dict = self.head(embeddings, labels=labels, label_weights=loss_weight,
                                                 one_hot_labels=True)
            features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]

        else:
            raise Exception()

        return features_list, loss_dict

    def get_embeddings(self, inputs):
        return self.backbone(inputs)




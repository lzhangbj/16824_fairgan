import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.models.backbone import build_backbone
from balface.models.head import build_head

class RSCRecognizer(nn.Module):
    def __init__(self, model_cfg):
        super(RSCRecognizer, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.head = build_head(model_cfg.head)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.use_weight = model_cfg.use_weight

    def forward(self, inputs, labels=None, loss_weight=None, flag=False, epoch=0):
        x = self.backbone(inputs)

        # rsc training
        if flag and self.training:
            interval = 10000
            if epoch % interval == 0:
                self.pecent = 1.0 / 10 + (epoch / interval) * 2.0 / 10

            self.eval()
            x_new = x.clone().detach()
            x_new.requires_grad=True
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = getattr(self.head, 'classifier_1').compute_cls_logits(x_new_view, no_grad=False)
            class_num = output.shape[1]
            index = labels[:, 0]
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot.requires_grad = False
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse.requires_grad = False
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = math.ceil(HW * 1 / 3.0)
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = getattr(self.head, 'classifier_1').compute_cls_logits(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][
                int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all.requires_grad=True
            x = x * mask_all

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        label_weights = None
        # if self.use_weight:
        #     label_weights = torch.tensor([1., 10, 10, 10, 10, 10, 10.]).float().cuda()

        features_list, loss_dict = self.head(x, labels=labels, label_weights=loss_weight)

        return features_list, loss_dict

    def get_embeddings(self, inputs):
        return self.avgpool(self.backbone(inputs)).view(input.size(0, -1))




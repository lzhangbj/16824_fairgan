import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balface.models.backbone import build_backbone
from balface.models.feat_fuser import build_feat_fuser
from balface.models.head import build_head
from balface.utils import sync_tensor_across_gpus

class LatentaugRecognizer(nn.Module):
	def __init__(self, model_cfg):
		super(LatentaugRecognizer, self).__init__()
		self.backbone = build_backbone(model_cfg.backbone)
		self.latent_encoder = build_backbone(model_cfg.latent_encoder)
		self.feat_fuser = build_feat_fuser(model_cfg.feat_fuser)
		self.head = build_head(model_cfg.head)
		
		self.use_weight = model_cfg.use_weight
		self.feat_aug = model_cfg.feat_aug
		self.n_class = model_cfg.head.n_classes[0]
		self.aug_num_per_gpu = model_cfg.aug_num_per_gpu
		aug_ratio = np.array(model_cfg.aug_ratio).astype(np.float32)
		self.aug_ratio = aug_ratio / aug_ratio.sum()
		self.blend_alpha = model_cfg.blend_alpha
		# self.feat_norm = model_cfg.feat_norm
		# self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

	def forward(self, inputs, labels=None, loss_weight=None, one_hot_labels=False):
		embeddings = self.backbone(inputs)
		# h, w = embeddings.size()[2:]
		
		with torch.no_grad():
			latents = self.latent_encoder(inputs)

		assert not one_hot_labels

		if self.feat_aug == 'none' or not self.training:
			# embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
			agg_feat = self.feat_fuser(embeddings, latents)
			features_list, loss_dict = self.head(agg_feat, labels=labels, label_weights=loss_weight, one_hot_labels=one_hot_labels)

		elif self.feat_aug == 'pre-aug':
			# embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
			global_embeddings = sync_tensor_across_gpus(embeddings)
			global_latents = sync_tensor_across_gpus(latents)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

			sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings1 = global_embeddings[sampled1_indices]
			sampled_latents1 = global_latents[sampled1_indices]
			sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class)

			sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings2 = global_embeddings[sampled2_indices]
			sampled_latents2 = global_latents[sampled2_indices]
			sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)

			blend_alphas = torch.tensor(np.random.uniform(0., self.blend_alpha, size=self.aug_num_per_gpu)).float().cuda().unsqueeze(1)

			aug_embeddings = sampled_embeddings2 * blend_alphas + (1-blend_alphas) * sampled_embeddings1
			aug_latents = sampled_latents2 * blend_alphas + (1 - blend_alphas) * sampled_latents1
			aug_targets = sampled_targets1 * (1-blend_alphas) + sampled_targets2 * blend_alphas

			embeddings = torch.cat([embeddings, aug_embeddings])
			latents = torch.cat([latents, aug_latents])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

			agg_feats = self.feat_fuser(embeddings, latents)

			features_list, loss_dict = self.head(agg_feats, labels=labels, label_weights=loss_weight,
												 one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]

		elif self.feat_aug == 'pre-aug-dom':
			# embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
			global_embeddings = sync_tensor_across_gpus(embeddings)
			global_latents = sync_tensor_across_gpus(latents)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

			sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings1 = global_embeddings[sampled1_indices]
			sampled_latents1 = global_latents[sampled1_indices]
			sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class).float()

			rev_aug_ratio = 1./self.aug_ratio
			rev_aug_ratio = rev_aug_ratio / rev_aug_ratio.sum()
			sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=rev_aug_ratio)
			sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_embeddings2 = global_embeddings[sampled2_indices]
			sampled_latents2 = global_latents[sampled2_indices]
			# sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)

			blend_alphas = torch.tensor(np.random.uniform(0., self.blend_alpha, size=self.aug_num_per_gpu)).float().cuda().unsqueeze(1)

			aug_embeddings = sampled_embeddings2 * blend_alphas + (1-blend_alphas) * sampled_embeddings1
			aug_latents = sampled_latents2 * blend_alphas + (1 - blend_alphas) * sampled_latents1
			aug_targets = sampled_targets1 #* (1-blend_alphas) + sampled_targets2 * blend_alphas

			embeddings = torch.cat([embeddings, aug_embeddings])
			latents = torch.cat([latents, aug_latents])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

			agg_feats = self.feat_fuser(embeddings, latents)

			features_list, loss_dict = self.head(agg_feats, labels=labels, label_weights=loss_weight,
												 one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]


		elif self.feat_aug == 'post-aug':
			# embeddings = self.avg_pool(embeddings).view(embeddings.size(0), embeddings.size(1))
			agg_feats = self.feat_fuser(embeddings, latents)

			global_agg_feats = sync_tensor_across_gpus(agg_feats)
			global_targets = sync_tensor_across_gpus(labels[:, 0].contiguous())
			B = global_targets.size(0)
			cls_indices_list = [torch.nonzero(global_targets==i).squeeze(1).cpu().numpy() for i in range(self.n_class)]

			sampled_cls1 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls1_cls_num_list = [np.sum(sampled_cls1==i, dtype=np.int32) for i in range(self.n_class)]
			sampled1_indices = np.concatenate([np.random.choice(cls_indices_list[i], sampled_cls1_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_agg_feats1 = global_agg_feats[sampled1_indices]
			sampled_targets1 = F.one_hot(global_targets[sampled1_indices], num_classes=self.n_class)

			sampled_cls2 = np.random.choice(np.arange(self.n_class, dtype=np.int32), self.aug_num_per_gpu, replace=True, p=self.aug_ratio)
			sampled_cls2_cls_num_list = [np.sum(sampled_cls2 == i, dtype=np.int32) for i in range(self.n_class)]
			sampled2_indices = np.concatenate(
				[np.random.choice(cls_indices_list[i], sampled_cls2_cls_num_list[i], replace=True) for i in range(self.n_class)])
			sampled_agg_feats2 = global_agg_feats[sampled2_indices]
			sampled_targets2 = F.one_hot(global_targets[sampled2_indices], num_classes=self.n_class)

			blend_alphas = torch.tensor(np.random.uniform(0., self.blend_alpha, size=self.aug_num_per_gpu)).float().cuda().unsqueeze(1)

			aug_agg_feats = sampled_agg_feats2 * blend_alphas + (1-blend_alphas) * sampled_agg_feats1
			aug_targets = sampled_targets1 * (1-blend_alphas) + sampled_targets2 * blend_alphas

			agg_feats = torch.cat([agg_feats, aug_agg_feats])
			labels = torch.cat([F.one_hot(labels[:, 0], num_classes=self.n_class), aug_targets]).unsqueeze(1)

			features_list, loss_dict = self.head(agg_feats, labels=labels, label_weights=loss_weight,
												 one_hot_labels=True)
			features_list = [feature[:-self.aug_num_per_gpu] for feature in features_list]
	  
		else:
			raise Exception()

		return features_list, loss_dict

	def get_embeddings(self, inputs):
		return self.backbone(inputs)

	@torch.no_grad()
	def get_latents(self, inputs):
		return self.latent_encoder(inputs)
	
	def get_fused_feat(self, inputs):
		embeddings = self.backbone(inputs)
		latents = self.latent_encoder(inputs)
		return self.feat_fuser(embeddings, latents)





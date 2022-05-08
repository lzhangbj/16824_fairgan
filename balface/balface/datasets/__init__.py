from .cls_datasets import *
from .unlabel_datasets import *
from .benchmark import *


def build_dataset(dataset_cfg):
	name = dataset_cfg.dataset
	if name == 'cls_fairface':
		return CLSFairFaceDataset(**dataset_cfg)
	elif name == 'cls_fairface_4races':
		return CLSFairFace4RacesDataset(**dataset_cfg)
	elif name == 'cls_fairface_4races_025':
		return CLSFairFace4Races025Dataset(**dataset_cfg)
	elif name == 'cls_gan_fairface_4races_025':
		return CLSGANFairFace4Races025Dataset(**dataset_cfg)
	elif name == 'cls_fairface_4races_025_aug':
		return CLSFairFace4Races025augDataset(**dataset_cfg)
	elif name == 'cls_fairface_gender_025':
		return CLSFairFaceGender025Dataset(**dataset_cfg)
	elif name == 'cls_fairface_gender_025_aug':
		return CLSFairFaceGender025augDataset(**dataset_cfg)
	elif name == 'cls_fairface_age_025':
		return CLSFairFaceAge025Dataset(**dataset_cfg)
	elif name == 'cls_fairface_age_025_aug':
		return CLSFairFaceAge025augDataset(**dataset_cfg)
	elif name == 'cls_fairface_4races_025_aug_denoise':
		return CLSFairFace4Races025augDenoiseDataset(**dataset_cfg)
	elif name == 'cls_fairface_4races_025_ratioaug':
		return CLSFairFace4Races025ratioaugDataset(**dataset_cfg)
	elif name == 'cls_utkface_race':
		return CLSUTKFaceRacesDataset(**dataset_cfg)
	elif name == 'cls_utkface_race_aug':
		return CLSUTKFaceRacesAugDataset(**dataset_cfg)
	elif name == 'cls_utkface_gender':
		return CLSUTKFaceGenderDataset(**dataset_cfg)
	elif name == 'cls_utkface_gender_aug':
		return CLSUTKFaceGenderAugDataset(**dataset_cfg)
	elif name == 'cutmix_fairface_4races':
		return CutmixFairFace4RacesDataset(**dataset_cfg)
	elif name == 'cls_fairface_5races':
		return CLSFairFace5RacesDataset(**dataset_cfg)
	elif name == 'cls_fairface_6races':
		return CLSFairFace6RacesDataset(**dataset_cfg)
	elif name == 'cls_fairface_7races':
		return CLSFairFace7RacesDataset(**dataset_cfg)
	elif name == 'ssp_base':
		return SSPBaseDataset(**dataset_cfg)
	elif name == 'unlabel':
		return UnlabelDataset(**dataset_cfg)
	elif name == 'fixmatch':
		return FixMatchDataset(**dataset_cfg)
	elif name == 'fixmatch_025':
		return FixMatch025Dataset(**dataset_cfg)
	elif name == 'imbalance_cifar_10':
		return IMBALANCECIFAR10(**dataset_cfg)
	elif name == 'imbalance_cifar_100':
		return IMBALANCECIFAR100(**dataset_cfg)
	elif name == 'cifar_100':
		return CIFAR100(**dataset_cfg)
	elif name == 'cifar_100':
		return CIFAR100(**dataset_cfg)
	else:
		raise Exception()


__all__ = [
	'build_dataset'
]


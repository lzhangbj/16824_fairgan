from .clsfairface_dataset import CLSFairFaceDataset, CLSFairFace4RacesDataset, \
	CLSFairFace6RacesDataset, CLSFairFace5RacesDataset, CLSFairFace7RacesDataset, \
	CLSFairFace4Races025augDataset, CLSFairFace4Races025Dataset, CLSFairFace4Races025ratioaugDataset, \
	CLSFairFace4Races025augDenoiseDataset

from .cls_gender_fairface_dataset import CLSFairFaceGender025augDataset, CLSFairFaceGender025Dataset
from .cls_age_fairface_dataset import CLSFairFaceAge025augDataset, CLSFairFaceAge025Dataset

from .cutmix_dataset import CutmixFairFace4RacesDataset
from .clsganfairface_dataset import CLSGANFairFace4Races025Dataset
from .cls_utkface_race_dataset import CLSUTKFaceRacesDataset, CLSUTKFaceRacesAugDataset
from .cls_utkface_gender_dataset import CLSUTKFaceGenderAugDataset, CLSUTKFaceGenderDataset

__all__ = [
    'CLSFairFaceDataset', 'CLSFairFace4RacesDataset', 'CLSFairFace6RacesDataset', 'CLSFairFace5RacesDataset', 'CLSFairFace7RacesDataset',
    'CutmixFairFace4RacesDataset', 'CLSFairFace4Races025Dataset', 'CLSFairFace4Races025augDataset', 'CLSFairFace4Races025ratioaugDataset',
	'CLSGANFairFace4Races025Dataset', 'CLSFairFace4Races025augDenoiseDataset',
	'CLSFairFaceAge025Dataset', 'CLSFairFaceAge025augDataset',
	'CLSFairFaceGender025Dataset', 'CLSFairFaceGender025augDataset',
	'CLSUTKFaceRacesDataset', 'CLSUTKFaceRacesAugDataset',
	'CLSUTKFaceGenderAugDataset', 'CLSUTKFaceGenderDataset'
]
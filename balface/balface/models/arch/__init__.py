import torch.distributed as dist

from .cls_recognizer import CLSRecognizer
from .ssl_recognizer import SSLRecognizer
from .ssp_recognizer import SSPRecognizer
from .fixmatch_recognizer import FixMatchRecognizer
from .cluster_recognizer import ClusterRecognizer
from .ssl_ssp_recognizer import SSLSSPRecognizer
from .rsc_recognizer import RSCRecognizer
from .benchmark_recognizer import BenchmarkRecognizer
from .feataug_recognizer import FeataugRecognizer
from .latent_aug import LatentaugRecognizer
from .cls_featmix_recognizer import CLSFeatMixRecognizer


def build_model(model_cfg):
    if model_cfg.name == 'cls_recognizer':
        model =  CLSRecognizer(model_cfg)
    elif model_cfg.name == 'cls_featmix_recognizer':
        model =  CLSFeatMixRecognizer(model_cfg)
    elif model_cfg.name == 'benchmark_recognizer':
        model =  BenchmarkRecognizer(model_cfg)
    elif model_cfg.name == 'ssl_recognizer':
        model =  SSLRecognizer(model_cfg)
    elif model_cfg.name == 'ssp_recognizer':
        model =  SSPRecognizer(model_cfg)
    elif model_cfg.name == 'fixmatch_recognizer':
        model =  FixMatchRecognizer(model_cfg)
    elif model_cfg.name == 'cluster_recognizer':
        model =  ClusterRecognizer(model_cfg)
    elif model_cfg.name == 'ssl_ssp_recognizer':
        model =  SSLSSPRecognizer(model_cfg)
    elif model_cfg.name == 'rsc_recognizer':
        model =  RSCRecognizer(model_cfg)
    elif model_cfg.name == 'feataug_recognizer':
        model =  FeataugRecognizer(model_cfg)
    elif model_cfg.name == 'latentaug_recognizer':
        model =  LatentaugRecognizer(model_cfg)
    else:
        raise Exception()
    
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f" whole arch parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model


__all__ = [
    'CLSRecognizer', 'SSLRecognizer', 'SSPRecognizer', 'SSLSSPRecognizer', 'build_model', 'FixMatchRecognizer',
    'ClusterRecognizer', 'SSLSSPRecognizer', 'BenchmarkRecognizer', 'FeataugRecognizer', 'LatentaugRecognizer',
	'CLSFeatMixRecognizer'
]

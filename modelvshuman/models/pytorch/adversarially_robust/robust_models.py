# Adapted from https://github.com/microsoft/robust-models-transfer by Santiago Cadena
#     @InProceedings{salman2020adversarially,
#         title={Do Adversarially Robust ImageNet Models Transfer Better?},
#         author={Hadi Salman and Andrew Ilyas and Logan Engstrom and Ashish Kapoor and Aleksander Madry},
#         year={2020},
#         booktitle={ArXiv preprint arXiv:2007.08489}
#     }

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

__all__ = ['resnet50_l2_eps0', 'resnet50_l2_eps0_01', 'resnet50_l2_eps0_03',
           'resnet50_l2_eps0_05', 'resnet50_l2_eps0_1', 'resnet50_l2_eps0_25',
           'resnet50_l2_eps0_5', 'resnet50_l2_eps1', 'resnet50_l2_eps3', 'resnet50_l2_eps5']

model_urls = {"resnet50_l2_eps0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.ckpt",
              "resnet50_l2_eps0_01": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.01.ckpt",
              "resnet50_l2_eps0_03": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.03.ckpt",
              "resnet50_l2_eps0_05": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.05.ckpt",
              "resnet50_l2_eps0_1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.1.ckpt",
              "resnet50_l2_eps0_25": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.25.ckpt",
              "resnet50_l2_eps0_5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.5.ckpt",
              "resnet50_l2_eps1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps1.ckpt",
              "resnet50_l2_eps3": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt",
              "resnet50_l2_eps5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps5.ckpt"}


def _model(arch, model_fn, pretrained, progress, use_data_parallel, **kwargs):
    
    model = model_fn(pretrained=False)   

    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[arch], progress=progress)
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items()\
              if k[:len('module.model.')] == 'module.model.'}  # Consider only the model and not normalizers or attacker
        model.load_state_dict(sd)
        
    model = torch.nn.DataParallel(model) if use_data_parallel else model 
        
    return model

              
def resnet50_l2_eps0(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0  L2-robustness on ImageNet. Accuracy: 75.80. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_01(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.01  L2-robustness on ImageNet. Accuracy: 75.68. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_01', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_03(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.03  L2-robustness on ImageNet. Accuracy: 75.76. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_03', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_05(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.05  L2-robustness on ImageNet. Accuracy: 75.59. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_05', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_1(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.1  L2-robustness on ImageNet. Accuracy: 74.78. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_1', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_25(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.25  L2-robustness on ImageNet. Accuracy: 74.14. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_25', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_5(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.5  L2-robustness on ImageNet. Accuracy: 73.16. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_5', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps1(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 1.0  L2-robustness on ImageNet. Accuracy: 70.43. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps1', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps3(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 3.0  L2-robustness on ImageNet. Accuracy: 62.83. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps3', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps5(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 5.0  L2-robustness on ImageNet. Accuracy: 56.13. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps5', resnet50, pretrained, progress, use_data_parallel, **kwargs)

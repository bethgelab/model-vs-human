import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils import model_zoo


CLASSIFIER_WEIGHTS = {
    "InsDis": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/InsDis_classifier.pth",
    "CMC": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/CMC_classifier.pth",
    "MoCo": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/MoCo_classifier.pth",
    "PIRL": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/PIRL_classifier.pth",
    "MoCoV2": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/MoCov2_classifier.pth",
    "InfoMin": "https://github.com/rgeirhos/model-vs-human/releases/download/v0.3/InfoMin_classifier.pth"
}

PYCONTRAST_CLASSIFIER_WEIGHTS_MLCLOUD = {
    "InsDis": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/InsDis_classifier.pth",
    "CMC": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/CMC_classifier.pth",
    "MoCo": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/MoCo_classifier.pth",
    "PIRL": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/PIRL_classifier.pth",
    "MoCoV2": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/MoCov2_classifier.pth",
    "InfoMin": "/mnt/qb/bethge/knarayanappa/unsup_resnet50/InfoMin_classifier.pth"
}

PYCONTRAST_CLASSIFIER_WEIGHTS_GPFS = {
    "InsDis": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/InsDis_classifier.pth",
    "CMC": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/CMC_classifier.pth",
    "MoCo": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/MoCo_classifier.pth",
    "MoCoV2": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/MoCov2_classifier.pth",
    "PIRL": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/PIRL_classifier.pth",
    "InfoMin": "/gpfs01/bethge/share/modelvshuman_models/unsup_resnet50/InfoMin_classifier.pth"
}


PYTORCH_HUB = "kantharajucn/PyContrast:push_pretrained_models_to_pytorch_hub"


def build_classifier(model_name, classes=1000):
    """Building a linear layer"""

    n_class = classes

    n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    if os.path.exists("/mnt/qb/bethge/"):
        checkpoint = torch.load(PYCONTRAST_CLASSIFIER_WEIGHTS_MLCLOUD.get(model_name), map_location='cpu')
    elif os.path.exists("/gpfs01/bethge/share/"):
        checkpoint = torch.load(PYCONTRAST_CLASSIFIER_WEIGHTS_GPFS.get(model_name), map_location='cpu')
    else:
        raise ValueError("classifier weights not found")
    state_dict = OrderedDict()
    for k, v in checkpoint["classifier"].items():
        k = k.replace('module.', '')
        state_dict[k] = v
    classifier.load_state_dict(state_dict)

    return classifier


def InsDis(pretrained=False, **kwargs):
    """
    Unsupervised Feature Learning via Non-parameteric Instance Discrimination
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InsDis"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)

    return model, classifier


def CMC(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "CMC"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def MoCo(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCo"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def MoCoV2(pretrained=False, **kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCoV2"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def PIRL(pretrained=False, **kwargs):
    """
    Self-Supervised Learning of Pretext-Invariant Representations
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "PIRL"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InfoMin"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier

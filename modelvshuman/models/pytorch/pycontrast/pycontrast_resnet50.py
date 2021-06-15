from collections import OrderedDict

import torch
import torch.nn as nn

CLASSIFIER_WEIGHTS = {
    "InsDis": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InsDis_classifier.pth",
    "CMC": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/CMC_classifier.pth",
    "MoCo": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCo_classifier.pth",
    "PIRL": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/PIRL_classifier.pth",
    "MoCoV2": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCov2_classifier.pth",
    "InfoMin": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InfoMin_classifier.pth"
}


PYTORCH_HUB = "kantharajucn/PyContrast:push_pretrained_models_to_pytorch_hub"


def build_classifier(model_name, classes=1000):
    """Building a linear layer"""

    n_class = classes

    n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    checkpoint = torch.hub.load_state_dict_from_url(CLASSIFIER_WEIGHTS.get(model_name), map_location='cpu')

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

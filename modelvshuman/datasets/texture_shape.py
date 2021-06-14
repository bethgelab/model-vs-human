from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from . import decision_mappings, info_mappings
from .base import Dataset
from .dataloaders import PytorchLoader
from .registry import register_dataset
from .. import constants as c
from ..evaluation import metrics as m

__all__ = ["original", "greyscale", "texture", "edge", "silhouette",
           "cue_conflict"]


@dataclass
class TextureShapeParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


def _get_dataset(name, *args, **kwargs):
    params = TextureShapeParams(path=pjoin(c.DATASET_DIR, name))
    return Dataset(name=name,
                   params=params,
                   loader=PytorchLoader,
                   *args,
                   **kwargs)


@register_dataset(name="original")
def original(*args, **kwargs):
    return _get_dataset(name="original", *args, **kwargs)


@register_dataset(name="greyscale")
def greyscale(*args, **kwargs):
    return _get_dataset(name="greyscale", *args, **kwargs)


@register_dataset(name="texture")
def texture(*args, **kwargs):
    return _get_dataset(name="texture", *args, **kwargs)


@register_dataset(name="edge")
def edge(*args, **kwargs):
    return _get_dataset(name="edge", *args, **kwargs)


@register_dataset(name="silhouette")
def silhouette(*args, **kwargs):
    return _get_dataset("silhouette", *args, **kwargs)


@register_dataset(name="cue-conflict")
def cue_conflict(*args, **kwargs):
    return _get_dataset("cue-conflict", *args, **kwargs)

from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from . import decision_mappings, info_mappings
from .base import Dataset
from .dataloaders import PytorchLoader
from .registry import register_dataset
from .. import constants as c
from ..evaluation import metrics as m


@dataclass
class ImageNetParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1), m.Accuracy(topk=5)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo1000ClassesMapping()
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


@register_dataset(name='imagenet_validation')
def imagenet_validation(*args, **kwargs):
    params = ImageNetParams(image_size=256,
                            path=pjoin(c.DATASET_DIR, "imagenet_validation"))
    return Dataset(name="imagenet_validation",
                   params=params,
                   loader=PytorchLoader,
                   *args,
                   **kwargs)

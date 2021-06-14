from os.path import join as pjoin

from .base import Dataset
from .dataloaders import PytorchLoader
from .imagenet import ImageNetParams
from .registry import register_dataset
from .. import constants as c
from . import info_mappings, decision_mappings


@register_dataset(name='stylized')
def stylized(*args, **kwargs):
    params = ImageNetParams(path=pjoin(c.DATASET_DIR, "stylized"),
                            decision_mapping=decision_mappings.ImageNetProbabilitiesTo16ClassesMapping(),
                            info_mapping=info_mappings.InfoMappingWithSessions(),
                            contains_sessions=True)
    return Dataset(name="stylized",
                   params=params,
                   loader=PytorchLoader,
                   *args,
                   **kwargs)

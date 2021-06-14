from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from .registry import register_dataset
from .. import constants as c
from . import decision_mappings, info_mappings
from .dataloaders import PytorchLoader
from ..evaluation import metrics as m

from .base import Dataset
from .experiments import *

__all__ = ["colour", "contrast", "high_pass", "low_pass",
           "phase_scrambling", "power_equalisation",
           "false_colour", "rotation", "eidolonI",
           "eidolonII", "eidolonIII", "uniform_noise"]


@dataclass
class NoiseGeneralisationParams:
    path: str = ""
    experiments: List = field(default_factory=list)
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    info_mapping: object = info_mappings.InfoMappingWithSessions()
    contains_sessions: bool = True


def _get_dataset(name, params, *args, **kwargs):
    assert params is not None, "Dataset params are missing"
    params.path = pjoin(c.DATASET_DIR, name)
    return Dataset(name=name,
                   params=params,
                   loader=PytorchLoader,
                   *args,
                   **kwargs)


@register_dataset(name="colour")
def colour(*args, **kwargs):
    return _get_dataset(name="colour",
                        params=NoiseGeneralisationParams(experiments=[colour_experiment]),
                        *args, **kwargs)


@register_dataset(name="contrast")
def contrast(*args, **kwargs):
    return _get_dataset(name="contrast",
                        params=NoiseGeneralisationParams(experiments=[contrast_experiment]),
                        *args, **kwargs)


@register_dataset(name="high-pass")
def high_pass(*args, **kwargs):
    return _get_dataset(name="high-pass",
                        params=NoiseGeneralisationParams(experiments=[high_pass_experiment]),
                        *args, **kwargs)


@register_dataset(name="low-pass")
def low_pass(*args, **kwargs):
    return _get_dataset(name="low-pass",
                        params=NoiseGeneralisationParams(experiments=[low_pass_experiment]),
                        *args, **kwargs)


@register_dataset(name="phase-scrambling")
def phase_scrambling(*args, **kwargs):
    return _get_dataset(name="phase-scrambling",
                        params=NoiseGeneralisationParams(experiments=[phase_scrambling_experiment]),
                        *args, **kwargs)


@register_dataset(name="power-equalisation")
def power_equalisation(*args, **kwargs):
    return _get_dataset(name="power-equalisation",
                        params=NoiseGeneralisationParams(experiments=[power_equalisation_experiment]),
                        *args, **kwargs)


@register_dataset(name="false-colour")
def false_colour(*args, **kwargs):
    return _get_dataset(name="false-colour",
                        params=NoiseGeneralisationParams(experiments=[false_colour_experiment]),
                        *args, **kwargs)


@register_dataset(name="rotation")
def rotation(*args, **kwargs):
    return _get_dataset(name="rotation",
                        params=NoiseGeneralisationParams(experiments=[rotation_experiment]),
                        *args, **kwargs)


@register_dataset(name="eidolonI")
def eidolonI(*args, **kwargs):
    return _get_dataset(name="eidolonI",
                        params=NoiseGeneralisationParams(experiments=[eidolonI_experiment]),
                        *args, **kwargs)


@register_dataset(name="eidolonII")
def eidolonII(*args, **kwargs):
    return _get_dataset(name="eidolonII",
                        params=NoiseGeneralisationParams(experiments=[eidolonII_experiment]),
                        *args, **kwargs)


@register_dataset(name="eidolonIII")
def eidolonIII(*args, **kwargs):
    return _get_dataset(name="eidolonIII",
                        params=NoiseGeneralisationParams(experiments=[eidolonIII_experiment]),
                        *args, **kwargs)


@register_dataset(name="uniform-noise")
def uniform_noise(*args, **kwargs):
    return _get_dataset(name="uniform-noise",
                        params=NoiseGeneralisationParams(experiments=[uniform_noise_experiment]),
                        *args, **kwargs)

"""
Generic evaluation functionality: evaluate on several datasets.
"""
from abc import ABC, abstractmethod

from .. import datasets
from ..helper import human_categories as hc
import numpy as np
import torch
import copy
from .. import constants as c
from ..datasets import info_mappings


class Metric(ABC):
    def __init__(self, name):
        self.name = name
        self.reset()

    def check_input(self, output, target, assert_ndarray=True):
        assert type(output) is np.ndarray
        assert len(output.shape) == 2, "output needs to have len(output.shape) == 2 instead of " + str(len(output.shape))

        if assert_ndarray:
            assert type(target) is np.ndarray
            assert output.shape[0] == target.shape[0]

    @abstractmethod
    def update(self, predictions, targets, paths):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def __str__(self):
        return "{}: {}".format(self.name, self.value)


class Accuracy(Metric):
    def __init__(self, name=None, topk=1):
        if name is None:
            name = "accuracy (top-{})".format(topk)
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, predictions, targets, paths):
        correct = [t in p[:self.topk] for t, p in zip(targets, predictions)]
        self._sum += np.sum(correct)
        self._count += len(predictions)

    @property
    def value(self):
        if self._count == 0:
            return 0
        return self._sum / self._count

    def __str__(self):
        return "{0:s}: {1:3.2f}".format(self.name, self.value * 100)


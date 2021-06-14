#!/usr/bin/env python

import numpy as np
from abc import ABC, abstractmethod

from ..helper import human_categories as hc
from ..helper import wordnet_functions as wnf


class DecisionMapping(ABC):
    def check_input(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    @abstractmethod
    def __call__(self, probabilities):
        pass


class ImageNetProbabilitiesTo1000ClassesMapping(DecisionMapping):
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self.categories = wnf.get_ilsvrc2012_WNIDs()

    def __call__(self, probabilities):
        self.check_input(probabilities)
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

class ImageNetProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.HumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

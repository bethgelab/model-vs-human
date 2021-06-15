from . import cli
from . import datasets
from . import evaluation
from . import models
from . import plotting
from .model_evaluator import ModelEvaluator
from .plotting.plot import plot
from .version import __version__, VERSION

Evaluate = ModelEvaluator
Plot = plot

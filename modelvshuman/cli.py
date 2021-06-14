#!/usr/bin/env python3

import logging

import click

from .datasets import list_datasets
from .models import list_models

logger = logging.getLogger(__name__)

supported_models = list_models("tensorflow") + list_models("pytorch")
supported_datasets = list_datasets()

print("")


@click.command()
@click.option("--models", "-m",
              type=click.Choice(supported_models, case_sensitive=True),
              multiple=True,
              required=True)
@click.option("--datasets", "-d",
              type=click.Choice(supported_datasets,
                                case_sensitive=True),
              multiple=True,
              required=True)
@click.option("--test-run", "-t",
              is_flag=True,
              help="If the test-run flag is set, results will not be saved to csv")
@click.option("--num-workers", "-w",
               type=int,
               default=30,
               help="Number of cpu workers for data loading")
@click.option("--batch-size", "-b",
              type=int,
              default=16,
              help="Batch size during evaluation")
@click.option("--print-predictions", "-p",
              type=bool,
              default=True,
              help="Print predictions")
def main(models, datasets, *args, **kwargs):
    """
    Entry point to the toolkit
    Returns:

    """
        
    from .model_evaluator import ModelEvaluator
    
    if "all" in models:
        models = supported_models
    if "all" in datasets:
        datasets = supported_datasets

    evaluate = ModelEvaluator()
    evaluate(models, datasets, *args, **kwargs)

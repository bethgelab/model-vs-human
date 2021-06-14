import logging
import os
import sys
import shutil
import requests
from os.path import join

import torchvision.models as zoomodels
from tqdm import tqdm

from . import constants as c
from . import datasets as dataset_module
from .models import pytorch_model_zoo, tensorflow_model_zoo, list_models

logger = logging.getLogger(__name__)
dataset_base_url = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{NAME}.tar.gz"


def try_download_dataset_from_github(dataset_name):
    download_url = dataset_base_url.format(NAME=dataset_name)
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        total_length = response.headers.get('content-length')
        if not os.path.exists(c.DATASET_DIR):
            os.makedirs(c.DATASET_DIR)
        dataset_file = join(c.DATASET_DIR, f'{dataset_name}.tar.gz')
        print(f"Downloading dataset {dataset_name} to {dataset_file}")
        with open(dataset_file, 'wb') as fd:
            if total_length is None:  # no content length header
                fd.write(response.content)
            else:
                for chunk in tqdm(response.iter_content(chunk_size=4096)):
                    fd.write(chunk)
        shutil.unpack_archive(dataset_file, c.DATASET_DIR)
        os.remove(dataset_file)
        return True
    else:
        return False


def load_dataset(name, *args, **kwargs):
    default_kwargs = {"batch_size": 16, "num_workers": 4}
    kwargs = {**default_kwargs, **kwargs}
    logger.info(f"Loading dataset {name}")
    supported_datasets = dataset_module.list_datasets()
    module_name = supported_datasets.get(name, None)
    if module_name is None:
        raise NameError(f"Dataset {name} is not supported, "
                        f"please select from {list(supported_datasets.keys())}")
    elif os.path.exists(join(c.DATASET_DIR, name)):
        return eval(f"dataset_module.{module_name}")(*args, **kwargs)
    elif try_download_dataset_from_github(name):
        return eval(f"dataset_module.{module_name}")(*args, **kwargs)
    else:
        raise NotImplementedError(f"Dataset {name} not available for download, please obtain the dataset "
                                  f"yourself and save it to {c.DATASET_DIR}")


def no_op():
    assert tensorflow_model_zoo
    assert pytorch_model_zoo


def load_model(model_name, *args):
    if model_name in zoomodels.__dict__:
        model = eval("pytorch_model_zoo.model_pytorch")(model_name, *args)
        framework = 'pytorch'
    elif model_name in list_models("pytorch"):
        model = eval(f"pytorch_model_zoo.{model_name}")(model_name, *args)
        framework = 'pytorch'
    elif model_name in list_models('tensorflow'):
        model = eval(f"tensorflow_model_zoo.{model_name}")(model_name, *args)
        framework = 'tensorflow'
    else:
        raise NameError(f"Model {model_name} is not supported.")
    return model, framework


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

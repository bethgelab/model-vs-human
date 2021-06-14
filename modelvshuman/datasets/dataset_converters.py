import tensorflow as tf
import torch
import numpy as np


class ToTensorflow(object):
    """This will actually convert the Pytorch Data loader into Tensorflow DataLoader"""

    def __init__(self, pytorch_loader):
        self.pytorch_loader = pytorch_loader
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def convert(self, x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x

    def __iter__(self):
        for images, *other in self.pytorch_loader:
            images = images.numpy().transpose([0, 2, 3, 1])  # tensorflow uses channel-last format
            images *= self.std
            images += self.mean
            images = tf.convert_to_tensor(images)
            other = (self.convert(x) for x in other)  # convert target to numpy
            yield (images, *other)

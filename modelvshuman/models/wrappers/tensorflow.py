import numpy as np
import tensorflow as tf
from skimage.transform import resize

tf.compat.v1.enable_eager_execution()
from .base import AbstractModel
from ...helper.human_categories import compute_imagenet_indices_for_category


def get_device(device=None):
    import tensorflow as tf

    if device is None:
        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
    if isinstance(device, str):
        device = tf.device(device)
    return device


class TensorflowModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args

    def softmax(self, logits):
        assert type(logits) is np.ndarray
        return tf.nn.softmax(logits).numpy()

    def forward_batch(self, images):
        device = get_device()
        with device:
            predictions = self.model(images)
            return predictions.numpy()

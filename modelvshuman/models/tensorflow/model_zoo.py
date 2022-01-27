from ..registry import register_model

from ..wrappers.tensorflow import TensorflowModel
from .build_model import build_model_from_hub


@register_model("tensorflow")
def efficientnet_b0(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def resnet50(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def mobilenet_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)


@register_model("tensorflow")
def inception_v1(model_name, *args):
    model = build_model_from_hub(model_name)
    return TensorflowModel(model, model_name, *args)

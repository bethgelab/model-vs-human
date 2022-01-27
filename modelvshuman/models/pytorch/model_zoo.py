#!/usr/bin/env python3
import torch

from ..registry import register_model
from ..wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel, \
    ViTPytorchModel, EfficientNetPytorchModel, SwagPytorchModel

_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"

_EFFICIENTNET_MODELS = "rwightman/gen-efficientnet-pytorch"


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet9(model_name, *args):
    from .bagnets.pytorchnet import bagnet9
    model = bagnet9(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet17(model_name, *args):
    from .bagnets.pytorchnet import bagnet17
    model = bagnet17(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet33(model_name, *args):
    from .bagnets.pytorchnet import bagnet33
    model = bagnet33(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x1_supervised_baseline
    model = simclr_resnet50x1_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x4_supervised_baseline
    model = simclr_resnet50x4_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1(model_name, *args):
    from .simclr import simclr_resnet50x1
    model = simclr_resnet50x1(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x2(model_name, *args):
    from .simclr import simclr_resnet50x2
    model = simclr_resnet50x2(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4(model_name, *args):
    from .simclr import simclr_resnet50x4
    model = simclr_resnet50x4(pretrained=True,
                              use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def InsDis(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InsDis
    model, classifier = InsDis(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCo(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCo
    model, classifier = MoCo(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCoV2(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCoV2
    model, classifier = MoCoV2(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def PIRL(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import PIRL
    model, classifier = PIRL(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def InfoMin(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InfoMin
    model, classifier = InfoMin(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0
    model = resnet50_l2_eps0()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_01(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_01
    model = resnet50_l2_eps0_01()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_03(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_03
    model = resnet50_l2_eps0_03()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_05(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_05
    model = resnet50_l2_eps0_05()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_1
    model = resnet50_l2_eps0_1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_25(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_25
    model = resnet50_l2_eps0_25()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_5
    model = resnet50_l2_eps0_5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps1
    model = resnet50_l2_eps1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps3(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps3
    model = resnet50_l2_eps3()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps5
    model = resnet50_l2_eps5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_es(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0_noisy_student(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "tf_efficientnet_b0_ns",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_l2_noisy_student_475(model_name, *args):
    model = torch.hub.load(_EFFICIENTNET_MODELS,
                           "tf_efficientnet_l2_ns_475",
                           pretrained=True)
    return EfficientNetPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_small_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_base_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_large_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def cspresnet50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspresnext50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspdarknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def darknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn92(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn98(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn131(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn107(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small_v2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w30(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w40(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w44(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w48(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w64(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls84(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def clip(model_name, *args):
    import clip
    model, _ = clip.load("ViT-B/32")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def clipRN50(model_name, *args):
    import clip
    model, _ = clip.load("RN50")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnet50_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def ResNeXt101_32x16d_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext101_32x16d_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x2_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x4(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x4_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_hard_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_hard_labels.pth",map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_soft_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_soft_labels.pth", map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def swag_regnety_16gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_16gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_32gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_32gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_128gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_128gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_b16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_l16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
    return SwagPytorchModel(model, model_name, input_size=512, *args)


@register_model("pytorch")
def swag_vit_h14_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
    return SwagPytorchModel(model, model_name, input_size=518, *args)

import math
import PIL
import clip
import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

from .base import AbstractModel
from ..pytorch.clip.imagenet_classes import imagenet_classes
from ..pytorch.clip.imagenet_templates import imagenet_templates


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

    return images


class PytorchModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.model.to(device())

    def to_numpy(self, x):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()

    def softmax(self, logits):
        assert type(logits) is np.ndarray

        softmax_op = torch.nn.Softmax(dim=1)
        softmax_output = softmax_op(torch.Tensor(logits))
        return self.to_numpy(softmax_output)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        self.model.eval()
        logits = self.model(images)
        return self.to_numpy(logits)


class PyContrastPytorchModel(PytorchModel):
    """
    This class inherits PytorchModel class to adapt model validation for Pycontrast pre-trained models from
    https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, model, classifier, model_name, *args):
        super(PyContrastPytorchModel, self).__init__(model, model_name, args)
        self.classifier = classifier
        self.classifier.to(device())

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        self.classifier.eval()
        feat = self.model(images, mode=2)
        output = self.classifier(feat)
        return self.to_numpy(output)


class ViTPytorchModel(PytorchModel):

    def __init__(self, model, model_name, img_size=(384, 384), *args):
        self.img_size = img_size
        super(ViTPytorchModel, self).__init__(model, model_name, args)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)

    def preprocess(self):
        # custom preprocessing from:
        # https://github.com/lukemelas/PyTorch-Pretrained-ViT

        return Compose([
            Resize(self.img_size),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])


class ClipPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super(ClipPytorchModel, self).__init__(model, model_name, *args)
        self.zeroshot_weights=self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
        
    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device())  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device())

        return zeroshot_weights

    def preprocess(self):
        n_px = self.model.visual.input_resolution
        return Compose([
            Resize(n_px, interpolation=PIL.Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        self.model.eval()
        
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)


class EfficientNetPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super(EfficientNetPytorchModel, self).__init__(model, model_name, *args)

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        img_size = 475
        crop_pct = 0.936
        scale_size = int(math.floor(img_size / crop_pct)) 
        return Compose([
            Resize(scale_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)


class SwagPytorchModel(PytorchModel):

    def __init__(self, model, model_name, input_size, *args):
        super(SwagPytorchModel, self).__init__(model, model_name, *args)
        self.input_size = input_size

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        return Compose([
            Resize(self.input_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(self.input_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())
        logits = self.model(images)
        return self.to_numpy(logits)    

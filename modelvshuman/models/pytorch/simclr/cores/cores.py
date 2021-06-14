from collections import OrderedDict, Iterable
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from ..utils.mlayer import clip_model, hook_model_module
import ptrnets


class Core:
    def initialize(self):
        raise NotImplementedError("Not initializing")

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda):
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
                

class TaskDrivenCore(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        model_name,
        layer_name,
        pretrained=True,
        bias=False,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs
    ):
        """
        Core from pretrained networks on image tasks.

        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            model_name (str): Name of the image recognition task model. Possible are all models in
            ptrnets: torchvision.models plus others
            layer_name (str): Name of the layer at which to clip the model
            pretrained (boolean): Whether to use a randomly initialized or pretrained network (default: True)
            bias (boolean): Whether to keep bias weights in the output layer (default: False)
            final_batchnorm (boolean): Whether to add a batch norm after the final conv layer (default: True)
            final_nonlinearity (boolean): Whether to add a final nonlinearity (ReLU) (default: True)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to freeze gradients of the core or to allow training
        """
        if kwargs:
            warnings.warn(
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__), UserWarning
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        model = getattr(ptrnets, model_name)(pretrained=pretrained)
        model_clipped = clip_model(model, layer_name)
        
        # Remove the bias of the last conv layer if not :bias:
        if not bias:
            if 'bias' in model_clipped[-1]._parameters:
                zeros = torch.zeros_like(model_clipped[-1].bias)
                model_clipped[-1].bias.data = zeros
        
        # Fix pretrained parameters during training
        if not fine_tune:
            for param in model_clipped.parameters():
                param.requires_grad = False
                

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module("TaskDriven", model_clipped)
        
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))
            
        print(self.features)

    def forward(self, input_):
        # If model is designed for RBG input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1 and self.features.TaskDriven[0].in_channels == 3:
            input_ = input_.repeat(1, 3, 1, 1)
        input_ = self.features(input_)
        return input_

    def regularizer(self):
        return 0   # useful for final loss

    
    @property
    def outchannels(self):
        """
        Function which returns the number of channels in the output conv layer. If the output layer is not a conv
        layer, the last conv layer in the network is used.

        Returns: Number of output channels
        """
        x = torch.randn(1,3,224,224)
        return self.features.TaskDriven(x).shape[1]

    def initialize(self, cuda=False):
        # Overwrite parent class's initialize function because initialization is done by the 'pretrained' parameter
        self.put_to_cuda(cuda=cuda)
        

        
class TaskDrivenCore2(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        model_name,
        layer_name,
        pretrained=True,
        bias=False,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs
    ):
        """
        Core from pretrained networks on image tasks.

        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            model_name (str): Name of the image recognition task model. Possible are all models in
            ptrnets: torchvision.models plus others
            layer_name (str): Name of the layer at which to clip the model
            pretrained (boolean): Whether to use a randomly initialized or pretrained network (default: True)
            bias (boolean): Whether to keep bias weights in the output layer (default: False)
            final_batchnorm (boolean): Whether to add a batch norm after the final conv layer (default: True)
            final_nonlinearity (boolean): Whether to add a final nonlinearity (ReLU) (default: True)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to freeze gradients of the core or to allow training
        """
        if kwargs:
            warnings.warn(
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__), UserWarning
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum
        self.use_probe = False
        self.layer_name = layer_name
        self.pretrained = pretrained

        # Download model and cut after specified layer
        self.model = getattr(ptrnets, model_name)(pretrained=pretrained)
        
        
        # Decide whether to probe the model with a forward hook or to clip the model by replicating architecture of the model up to layer :layer_name:
        x = torch.randn(1,3,224,224)
        try:
            model_clipped = clip_model(self.model, self.layer_name)
            clip_out = model_clipped(x);
        except:
            warnings.warn('Unable to clip model {} at layer {}. Using a probe instead'.format(model_name, self.layer_name))
            self.use_probe = True
        
        self.model_probe = self.probe_model() 
        
     
        if not(self.use_probe):
            if torch.allclose(self.model_probe(x), clip_out):
                warnings.warn('Unable to recover model outputs via a sequential modules. Using forward hook instead')
                self.use_probe = True
              
        
        # Remove the bias of the last conv layer if not :bias:
        if not bias and not self.use_probe:
            if 'bias' in model_clipped[-1]._parameters:
                zeros = torch.zeros_like(model_clipped[-1].bias)
                model_clipped[-1].bias.data = zeros
        
        # Fix pretrained parameters during training
        if not fine_tune and not self.use_probe:
            for param in model_clipped.parameters():
                param.requires_grad = False
                

        # Stack model modules
        self.features = nn.Sequential()
        
        if not(self.use_probe): self.features.add_module("TaskDriven", model_clipped)
        
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))
            
        # Remove model module if not(self.use_probe):
        
        if not(self.use_probe):
            del self.model

    def forward(self, input_):
        # If model is designed for RBG input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1: 
            input_ = input_.repeat(1, 3, 1, 1)   
        
        if self.use_probe:
            input_ = self.model_probe(input_)
        
        input_ = self.features(input_)
        return input_

    def regularizer(self):
        return 0   # useful for final loss

    
    def probe_model(self):
    
        assert self.layer_name in [n for n,_ in self.model.named_modules()], 'No module named {}'.format(self.layer_name)
        self.model.eval();
        hook = hook_model_module(self.model, self.layer_name)
        def func(x):
            try:
                self.model(x); 
            except:
                pass
            return hook(self.layer_name)

        return func
    
    @property
    def outchannels(self):
        """
        Function which returns the number of channels in the output conv layer. If the output layer is not a conv
        layer, the last conv layer in the network is used.

        Returns: Number of output channels
        """
        x = torch.randn(1,3,224,224)
        if self.use_probe:
            outch = self.model_probe(x).shape[1]
        else:
            outch = self.features.TaskDriven(x).shape[1]
        return outch

    def initialize(self, cuda=False):
        # Overwrite parent class's initialize function
        if not self.pretrained:
            self.apply(self.init_conv) 
        self.put_to_cuda(cuda=cuda)
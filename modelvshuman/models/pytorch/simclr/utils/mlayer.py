# Utils for getting model layers

from collections import OrderedDict
from torch import nn


def clip_model(model, layer_name):
    """
    Returns a copy of the model up to :layer_name:
    Params:
        model: (nn.Module instance)
        layer_name (string) Should be among the named modules of model
    Returns:
        clipped_model: (nn.Sequential) copy of model up to the layer
    Examples:
        from torchvision.models import resnet50, vgg19
        resnet = resnet50(pretrained=True)
        vgg    = vgg19(pretrained=True)
        clipped_resnet = clip_model(resnet, 'layer3.0.conv2')
        clipped_vgg    = clip_model(vgg, 'features.10')
    """
    
    assert layer_name in [n for n,_ in model.named_modules()], 'No module named {}'.format(layer_name)
    
    features = OrderedDict()
    nodes_iter = iter(layer_name.split('.'))
    
    def recursive(module, node = next(nodes_iter), prefix=[]):
        
        for name, layer in module.named_children():
            fullname = ".".join(prefix+[name])
            if (name == node) and (fullname != layer_name):
                recursive(layer, node = next(nodes_iter), prefix=[fullname])
                return
            else:
                features[name] = layer
                #print(fullname)
                if fullname == layer_name:
                    return

    recursive(model)
    
    clipped_model = nn.Sequential(features)
    
    return clipped_model


def probe_model(model, layer_name):
    
    assert layer_name in [n for n,_ in model.named_modules()], 'No module named {}'.format(layer_name)
    model.eval();
    #hook = hook_model(model)
    hook = hook_model_module(model, layer_name)
    def func(x):
        try:
            model(x); 
        except:
            pass
        return hook(layer_name)
    
    return func


class ModuleHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()



def hook_model(model):
    
    features = OrderedDict()
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features[".".join(prefix+[name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix+[name])
    hook_layers(model)

    def hook(layer):
        if layer == "labels":
            return list(features.values())[-1].features
        return features[layer].features
    return hook



def hook_model_module(model, module):
    
    features = OrderedDict()
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if ".".join(prefix+[name]) == module:
                    features[".".join(prefix+[name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix+[name])
    
    hook_layers(model)

    def hook(layer):
        if layer == "labels":
            return list(features.values())[-1].features
        return features[layer].features
    return hook

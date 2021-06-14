## Custom pytorch modules
import torch
from torch import nn

class Unnormalize(nn.Module):
    """
    Helper class for unnormalizing input tensor
    """
    def __init__(self, mean=[0], std=[1], inplace=False):
        super(Unnormalize, self).__init__()
        self.mean = mean
        self.std  = std
        self.inplace = inplace
    
    def forward(self, x):
        return unnormalize(x, self.mean, self.std, self.inplace)




def unnormalize(tensor, mean=[0], std=[1], inplace=False):
    """Unnormalize a tensor image by first multiplying by std (channel-wise) and then adding the mean (channel-wise)
    
    Args:
        tensor (Tensor): Tensor image of size (N, C, H, W) to be de-standarized.
        mean (sequence): Sequence of original means for each channel.
        std (sequence): Sequence of original standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalized Tensor image.
    
    """
    
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))
        
    if tensor.ndimension() != 4:
        raise ValueError('Expected tensor to be a tensor image of size (N, C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))
    if not inplace:
        tensor=tensor.clone()
    
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        
    if mean.ndim == 1:
        mean = mean[None, :, None, None]
    if std.ndim == 1:
        std = std[None, :, None, None]       
    
    tensor.mul_(std).add_(mean)
    return tensor
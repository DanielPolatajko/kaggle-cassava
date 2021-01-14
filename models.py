# Custom models used for Cassava competition


import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, _resnet
from typing import Type, Any, Union, List
import numpy as np

class SirenActivation(nn.Module):
    """
    Implements SIREN activation function as per "Implicit Neural Representations with Periodic
    Activation Functions", https://arxiv.org/pdf/2006.09661.pdf
    """

    def __init__(self):
        super(SirenActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class SirenBlock(BasicBlock):
    """
    Block for ResNet with SIREN activation
    """
    def __init__(self, *args, **kwargs):
        super(SirenBlock, self).__init__(*args, **kwargs)
        self.relu = SirenActivation()

class SirenBottleneck(Bottleneck):
    """
    Bottleneck lock for ResNet with SIREN activation
    """

    def __init__(self, *args, **kwargs):
        super(SirenBottleneck, self).__init__(*args, **kwargs)
        self.relu = SirenActivation()

class SirenResNet(ResNet):
    """
    ResNet builder with the SIREN activation layer
    """
    def __init__(self, *args, **kwargs):
        super(SirenResNet, self).__init__(*args, **kwargs)
        self.relu = SirenActivation()

# From torchvision.models.resnet, basically _resnet and resnext50_32x4d
def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = SirenResNet(block, layers, **kwargs)
    return model

def siren(**kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(SirenBottleneck, [3, 4, 6, 3],**kwargs)

def initialise_weights(network):
    """
    A utility function to initiliase the weights of a SIREN network according to the init scheme from the paper
    :param layer: The network layer to initiliase weights for (we use apply method to apply to network)
    :return:
    """

    def uniform_distribution(layer):
        """
        Does every layer except the first one, which is special
        :param layer: the layer upon which the function acts, settings weights
        :return:
        """
        classname = layer.__class__.__name__
        print(classname)
        if classname == "Conv2d":
            # get the number of the inputs
            n = layer.in_channels
            boundary = np.sqrt(6. / n)

            # deal with the first layer separately
            if n == 3:
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(
                        torch.nn.Parameter(torch.FloatTensor(layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]).uniform_(-boundary, boundary)))
                    layer.bias = torch.nn.Parameter(
                        torch.add(torch.zeros(layer.out_channels), 30))
            else:
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(torch.FloatTensor(layer.out_channels, layer.in_channels // layer.groups, layer.kernel_size[0], layer.kernel_size[1]).uniform_(-boundary, boundary))
                    layer.bias = torch.nn.Parameter(torch.zeros(layer.out_channels))

            # print(layer.weight)
            # print(layer.bias)

        elif classname == "Linear":
            # get the number of the inputs
            n = layer.in_features
            boundary = np.sqrt(6. / n)
            layer.weight.data.uniform_(-boundary, boundary)
            layer.bias.data.fill_(0)
            # print(layer.weight)
            # print(layer.bias)


    network.apply(uniform_distribution)




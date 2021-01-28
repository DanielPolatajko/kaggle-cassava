# Custom models used for Cassava competition


import torch

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, _resnet
from typing import Type, Any, Union, List
import numpy as np
import torch.nn.functional as F

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
    architecture: Any,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = architecture(block, layers, **kwargs)
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
    return _resnet(SirenResNet, SirenBottleneck, [3, 4, 6, 3],**kwargs)

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

class MishActivation(nn.Module):
    """
    Implements the Mish activation function as per https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
    """

    def __init__(self):
        super(MishActivation, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class MishBlock(BasicBlock):
    """
    Block for ResNet with Mish activation
    """
    def __init__(self, *args, **kwargs):
        super(MishBlock, self).__init__(*args, **kwargs)
        self.relu = MishActivation()

class MishBottleneck(Bottleneck):
    """
    Bottleneck lock for ResNet with Mish activation
    """

    def __init__(self, *args, **kwargs):
        super(MishBottleneck, self).__init__(*args, **kwargs)
        self.relu = MishActivation()

class MishResNet(ResNet):
    """
    ResNet builder with the Mish activation layer
    """
    def __init__(self, *args, **kwargs):
        super(MishResNet, self).__init__(*args, **kwargs)
        self.relu = SirenActivation()

def mishnet(**kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(MishResNet, MishBottleneck, [3, 4, 6, 3],**kwargs)


def replace_activations(model, activation):
    """
    Allow custom activation with pretrained Resnet model
    """
    model.relu = activation
    for module in model.layer1:
        module.relu = activation
    for module in model.layer2:
        module.relu = activation
    for module in model.layer3:
        module.relu = activation
    for module in model.layer4:
        module.relu = activation

    print(model)

    return None

class WSConv2d(nn.Conv2d):
    """
    Weight standardisation convolutional layer as per https://arxiv.org/abs/1903.10520
    Code from https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)





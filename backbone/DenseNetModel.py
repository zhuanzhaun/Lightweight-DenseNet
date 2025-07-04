import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor


class _DenseLayer(nn.Module):
    def __init__(self,
                 input_c: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False,
                 conv_kernel_size: int = 3):
        super(_DenseLayer, self).__init__()

        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=conv_kernel_size,
                                           stride=1,
                                           padding=conv_kernel_size//2,
                                           bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False,
                 conv_kernel_size: int = 3):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient,
                                conv_kernel_size=conv_kernel_size)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 pool_size: int = 0,
                 pool_stride: int = 0):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c,
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        if pool_size > 0:
            self.add_module("pool", nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride))


class DenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int] = (3, 3, 4),
                 num_init_features: int = 4,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 2,
                 memory_efficient: bool = False,
                 first_pool_size: int = 0,
                 first_pool_stride: int = 0,
                 second_pool_size: int = 0,
                 second_pool_stride: int = 0,
                 first_downsample: bool = False,
                 second_downsample: bool = False,
                 conv_kernel_size: int = 3):
        super(DenseNet, self).__init__()

        # first conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=3, 
                               stride=2 if first_downsample else 1, 
                               bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
        ]))

        if first_pool_size > 0:
            self.features.add_module("pool0", nn.AvgPool2d(kernel_size=first_pool_size, 
                                                          stride=first_pool_stride))

        # each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient,
                                conv_kernel_size=conv_kernel_size)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                pool_size = second_pool_size if i == 1 else 0
                pool_stride = second_pool_stride if i == 1 else 0
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2,
                                    pool_size=pool_size,
                                    pool_stride=pool_stride)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(**kwargs: Any) -> DenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=16,
                    block_config=(2, 2, 2),
                    num_init_features=4,
                    **kwargs)

def densenet121yuan(**kwargs: Any) -> DenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32),
                    num_init_features=64,
                    **kwargs)



def densenet169(**kwargs: Any) -> DenseNet:
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32),
                    num_init_features=64,
                    **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48),
                    num_init_features=64,
                    **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36),
                    num_init_features=96,
                    **kwargs)


def load_state_dict(model: nn.Module, weights_path: str) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(weights_path)

    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)
    print("successfully load pretrain-weights.")

def densenet_literature17(**kwargs: Any) -> DenseNet:
    """文献17模型：Densenet层：3,3,3.第一池化层4*4*1，第二池化层不需要。第一下采样，不需要。第二下采样，不需要。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 3),
                    num_init_features=4,
                    first_pool_size=4,
                    first_pool_stride=1,
                    **kwargs)

def densenet_literature20(**kwargs: Any) -> DenseNet:
    """文献20模型：Densenet层：2,4,2.第一池化层3*3*1，第二池化层2*2*2。第一下采样，不需要。第二下采样，不需要。"""
    return DenseNet(growth_rate=32,
                    block_config=(2, 4, 2),
                    num_init_features=4,
                    first_pool_size=3,
                    first_pool_stride=1,
                    second_pool_size=2,
                    second_pool_stride=2,
                    **kwargs)

def densenet_mod1(**kwargs: Any) -> DenseNet:
    """模型改1：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，需要。第二下采样，不需要。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    first_downsample=True,
                    **kwargs)

def densenet_mod2(**kwargs: Any) -> DenseNet:
    """模型改2：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，不需要。第二下采样，需要。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    second_downsample=True,
                    **kwargs)

def densenet_mod3(**kwargs: Any) -> DenseNet:
    """模型改3：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，需要。第二下采样，需要。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    first_downsample=True,
                    second_downsample=True,
                    **kwargs)

def densenet_mod4(**kwargs: Any) -> DenseNet:
    """模型改4：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，不需要。第二下采样，不需要，denseblock内部将3*3的卷积核改成5*5。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    conv_kernel_size=5,
                    **kwargs)

def densenet_mod5(**kwargs: Any) -> DenseNet:
    """模型改5：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，不需要。第二下采样，不需要，denseblock内部将3*3的卷积核改成7*7。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    conv_kernel_size=7,
                    **kwargs)

def densenet_mod6(**kwargs: Any) -> DenseNet:
    """模型改6：Densenet层：3,3,4.第一池化层不需要，第二池化层不需要。第一下采样，不需要。第二下采样，不需要，denseblock内部将3*3的卷积核改成3*3。"""
    return DenseNet(growth_rate=32,
                    block_config=(3, 3, 4),
                    num_init_features=4,
                    conv_kernel_size=3,
                    **kwargs)

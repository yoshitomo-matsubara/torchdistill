from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import register_model_func
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)
ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar10-wide_resnet40_4.pt',
    'cifar10-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar10-wide_resnet28_10.pt',
    'cifar10-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar10-wide_resnet16_8.pt',
    'cifar100-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar100-wide_resnet40_4.pt',
    'cifar100-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar100-wide_resnet28_10.pt',
    'cifar100-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar100-wide_resnet16_8.pt'
}


class WideBasicBlock(nn.Module):
    """
    A basic block of Wide ResNet for CIFAR datasets.

    :param in_planes: number of input feature planes.
    :type in_planes: int
    :param planes: number of output feature planes.
    :type planes: int
    :param dropout_rate: dropout rate.
    :type dropout_rate: float
    :param stride: stride for Conv2d.
    :type stride: int
    """
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet4Cifar(nn.Module):
    """
    Wide ResNet (WRN) model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    for CIFAR datasets, referring to https://github.com/szagoruyko/wide-residual-networks

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param depth: depth.
    :type depth: int
    :param k: widening factor.
    :type k: int
    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param block: block class.
    :type block: WideBasicBlock
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param norm_layer: normalization module class or callable object.
    :type norm_layer: typing.Callable or nn.Module or None
    """
    def __init__(self, depth, k, dropout_p, block, num_classes, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_wide_layer(block, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self._make_wide_layer(block, stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self._make_wide_layer(block, stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


@register_model_func
def wide_resnet(
        depth: int,
        k: int,
        dropout_p: float,
        num_classes: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    """
    Instantiates a Wide ResNet model for CIFAR datasets.

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param depth: depth.
    :type depth: int
    :param k: widening factor.
    :type k: int
    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: Wide ResNet model.
    :rtype: WideResNet4Cifar
    """
    assert (depth - 4) % 6 == 0, 'depth of Wide ResNet (WRN) should be 6n + 4'
    model = WideResNet4Cifar(depth, k, dropout_p, WideBasicBlock, num_classes, **kwargs)
    model_key = 'cifar{}-wide_resnet{}_{}'.format(num_classes, depth, k)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)
        model.load_state_dict(state_dict)
    elif pretrained:
        logger.warning(f'`pretrained` = True, but pretrained {model_key} model is not available')
    return model


@register_model_func
def wide_resnet40_4(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    WRN-40-4 model.

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: WRN-40-4 model.
    :rtype: WideResNet4Cifar
    """
    return wide_resnet(40, 4, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet28_10(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    WRN-28-10 model.

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: WRN-28-10 model.
    :rtype: WideResNet4Cifar
    """
    return wide_resnet(28, 10, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet16_8(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    WRN-16-8 model.

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: WRN-16-8 model.
    :rtype: WideResNet4Cifar
    """
    return wide_resnet(16, 8, dropout_p, num_classes, pretrained, progress, **kwargs)

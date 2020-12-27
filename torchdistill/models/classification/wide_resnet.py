from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from torchdistill.models.registry import register_model_func

"""
Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
for CIFAR datasets, referring to https://github.com/szagoruyko/wide-residual-networks
"""

MODEL_URL_DICT = {
    'cifar10-wide_resnet40_': '',
    'cifar10-wide_resnet28_10': '',
    'cifar10-wide_resnet16_': '',
    'cifar100-wide_resnet40_4': '',
    'cifar100-wide_resnet28_10': '',
    'cifar100-wide_resnet16_8': ''
}


class WideBasicBlock(nn.Module):
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


class WideResNet(nn.Module):
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
) -> WideResNet:
    assert (depth - 4) % 6 == 0, 'depth of Wide ResNet (WRN) should be 6n + 4'
    model = WideResNet(depth, k, dropout_p, WideBasicBlock, num_classes, **kwargs)
    model_key = 'cifar{}-wide_resnet{}_{}'.format(num_classes, depth, k)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)
        model.load_state_dict(state_dict)
    return model


@register_model_func
def wide_resnet40_4(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-40-4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p (float): p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(40, 4, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet28_10(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-28-10 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(28, 10, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet16_8(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-16-8 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(16, 8, dropout_p, num_classes, pretrained, progress, **kwargs)

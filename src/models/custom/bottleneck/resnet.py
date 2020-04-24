from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet152

from models.registry import register_class, register_func


@register_class
class Bottleneck4ResNet152(nn.Sequential):
    def __init__(self, bottleneck_channel=12):
        modules = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        ]
        super().__init__(*modules)


@register_class
class CustomResNet(nn.Sequential):
    def __init__(self, bottleneck, short_module_names, org_resnet):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        if 'fc' in short_module_set:
            short_module_set.remove('fc')

        for child_name, child_module in org_resnet.named_children():
            if child_name in short_module_set:
                if child_name == 'fc':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module

        super().__init__(module_dict)


@register_func
def custom_resnet152(bottleneck_channel=12, short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    bottleneck = Bottleneck4ResNet152(bottleneck_channel)
    org_model = resnet152(**kwargs)
    return CustomResNet(bottleneck, short_module_names, org_model)

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import densenet169, densenet201

from models.custom.bottleneck import register_bottleneck_class, register_bottleneck_func


@register_bottleneck_class
class Bottleneck4DenseNets(nn.Sequential):
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
            nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ]
        super().__init__(*modules)


@register_bottleneck_class
class CustomDenseNet(nn.Module):
    def __init__(self, bottleneck, short_feature_names, org_densenet):
        super().__init__()
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        ignored_set = set(short_feature_names).union({'classifier'})
        for child_name, child_module in org_densenet.named_children():
            if child_name not in ignored_set:
                module_dict[child_name] = child_module

        self.features = nn.Sequential(module_dict)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = org_densenet.classifier

    def forward(self, x):
        z = self.features(x)
        z = self.relu(z)
        z = self.adaptive_avgpool(z)
        z = torch.flatten(z, 1)
        return self.classifier(z)


@register_bottleneck_func
def custom_densenet169(bottleneck_channel=12, short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    bottleneck = Bottleneck4DenseNets(bottleneck_channel)
    org_model = densenet169(**kwargs)
    return CustomDenseNet(bottleneck, short_feature_names, org_model)


@register_bottleneck_func
def custom_densenet201(bottleneck_channel=12, short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    bottleneck = Bottleneck4DenseNets(bottleneck_channel)
    org_model = densenet201(**kwargs)
    return CustomDenseNet(bottleneck, short_feature_names, org_model)

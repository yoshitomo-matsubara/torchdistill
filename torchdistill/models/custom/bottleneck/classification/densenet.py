from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import densenet169, densenet201

from ..base import BottleneckBase
from ..registry import get_bottleneck_processor
from ....registry import register_model_class, register_model_func


@register_model_class
class Bottleneck4DenseNet(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """
    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
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
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


@register_model_class
class CustomDenseNet(nn.Module):
    def __init__(self, bottleneck, short_feature_names, org_densenet):
        super().__init__()
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_features_set = set(short_feature_names)
        if 'classifier' in short_features_set:
            short_features_set.remove('classifier')

        for child_name, child_module in org_densenet.features.named_children():
            if child_name in short_features_set:
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


@register_model_func
def custom_densenet169(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                       short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4DenseNet(bottleneck_channel, bottleneck_idx, compressor, decompressor)
    org_model = densenet169(**kwargs)
    return CustomDenseNet(bottleneck, short_feature_names, org_model)


@register_model_func
def custom_densenet201(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                       short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4DenseNet(bottleneck_channel, bottleneck_idx, compressor, decompressor)
    org_model = densenet201(**kwargs)
    return CustomDenseNet(bottleneck, short_feature_names, org_model)

from collections import OrderedDict

from torch import nn
from torchvision.models import inception_v3

from models.registry import register_class, register_func


@register_class
class Bottleneck4Inception3(nn.Sequential):
    def __init__(self, bottleneck_channel=12):
        modules = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        ]
        super().__init__(*modules)


@register_class
class CustomInception3(nn.Sequential):
    def __init__(self, bottleneck, short_module_names, org_resnet):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        child_name_list = list()
        for child_name, child_module in org_resnet.named_children():
            if child_name in short_module_set:
                if len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_2b_3x3' \
                        and child_name == 'Conv2d_3b_1x1':
                    module_dict['maxpool1'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool1')
                elif len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_4a_3x3' \
                        and child_name == 'Mixed_5b':
                    module_dict['maxpool2'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool2')
                elif child_name == 'fc':
                    module_dict['adaptive_avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
                    module_dict['dropout'] = nn.Dropout()
                    module_dict['flatten'] = nn.Flatten(1)

                module_dict[child_name] = child_module
                child_name_list.append(child_name)

        super().__init__(module_dict)


@register_func
def custom_inception_v3(bottleneck_channel=12, short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = [
            'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
            'Mixed_7a', 'Mixed_7b', 'Mixed_7c'
        ]

    bottleneck = Bottleneck4Inception3(bottleneck_channel)
    org_model = inception_v3(**kwargs)
    return CustomInception3(bottleneck, short_module_names, org_model)

from collections import OrderedDict
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.densenet import _DenseBlock, _Transition

from ..registry import register_model
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)
ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-densenet_bc_k12_depth100': ROOT_URL + '/v0.1.1/cifar10-densenet_bc_k12_depth100.pt',
    'cifar100-densenet_bc_k12_depth100': ROOT_URL + '/v0.1.1/cifar100-densenet_bc_k12_depth100.pt'
}


class DenseNet4Cifar(nn.Module):
    """
    DenseNet-BC model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    for CIFAR datasets, referring to https://github.com/liuzhuang13/DenseNet

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param growth_rate: number of filters to add each layer (`k` in paper).
    :type growth_rate: int
    :param block_config: three numbers of layers in each pooling block.
    :type block_config: list[int]
    :param num_init_features: number of filters to learn in the first convolution layer.
    :type num_init_features: int
    :param bn_size: multiplicative factor for number of bottleneck layers. (i.e. bn_size * k features in the bottleneck layer)
    :type bn_size: int
    :param drop_rate: dropout rate after each dense layer.
    :type drop_rate: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param memory_efficient: if True, uses checkpointing. Much more memory efficient, but slower. Refer to `"the paper" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ for details.
    :type memory_efficient: bool
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int] = (12, 12, 12),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 10,
        memory_efficient: bool = False
    ) -> None:

        super().__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
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


@register_model
def densenet(
    growth_rate: int,
    depth: int,
    num_init_features: int,
    bottleneck: bool,
    num_classes: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    """
    Instantiates a DenseNet model for CIFAR datasets.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param growth_rate: number of filters to add each layer (`k` in paper).
    :type growth_rate: int
    :param depth: depth.
    :type depth: int
    :param num_init_features: number of filters to learn in the first convolution layer.
    :type num_init_features: int
    :param bottleneck: if True, uses bottleneck.
    :type bottleneck: bool
    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: DenseNet model.
    :rtype: DenseNet4Cifar
    """

    n = (depth - 4) // 3
    if bottleneck:
        n //= 2
        num_init_features = growth_rate * 2
    model = DenseNet4Cifar(growth_rate, (n, n, n), num_init_features, num_classes=num_classes, **kwargs)
    base_model_name = 'densenet_bc' if bottleneck else 'densenet'
    model_key = 'cifar{}-{}_k{}_depth{}'.format(num_classes, base_model_name, growth_rate, depth)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)
        model.load_state_dict(state_dict)
    elif pretrained:
        logger.warning(f'`pretrained` = True, but pretrained {model_key} model is not available')
    return model


@register_model
def densenet_bc_k12_depth100(num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    DenseNet-BC (k=12, depth=100) model.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: DenseNet-BC (k=12, depth=100) model.
    :rtype: DenseNet4Cifar
    """
    return densenet(12, 100, 16, True, num_classes, pretrained, progress, **kwargs)


@register_model
def densenet_bc_k24_depth250(num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    DenseNet-BC (k=24, depth=250) model.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: DenseNet-BC (k=24, depth=250) model.
    :rtype: DenseNet4Cifar
    """
    return densenet(24, 250, 16, True, num_classes, pretrained, progress, **kwargs)


@register_model
def densenet_bc_k40_depth190(num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    DenseNet-BC (k=40, depth=190) model.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: DenseNet-BC (k=40, depth=190) model.
    :rtype: DenseNet4Cifar
    """
    return densenet(40, 190, 16, True, num_classes, pretrained, progress, **kwargs)

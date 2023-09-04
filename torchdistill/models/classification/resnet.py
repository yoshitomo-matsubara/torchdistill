from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, conv1x1

from ..registry import register_model_func
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)
ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-resnet20': ROOT_URL + '/v0.1.1/cifar10-resnet20.pt',
    'cifar10-resnet32': ROOT_URL + '/v0.1.1/cifar10-resnet32.pt',
    'cifar10-resnet44': ROOT_URL + '/v0.1.1/cifar10-resnet44.pt',
    'cifar10-resnet56': ROOT_URL + '/v0.1.1/cifar10-resnet56.pt',
    'cifar10-resnet110': ROOT_URL + '/v0.1.1/cifar10-resnet110.pt'
}


class ResNet4Cifar(nn.Module):
    """
    ResNet model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    for CIFAR datasets, referring to https://github.com/facebookarchive/fb.resnet.torch

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param block: block class.
    :type block: BasicBlock
    :param layers: three numbers of layers in each pooling block.
    :type layers: list[int]
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param zero_init_residual: if True, zero-initializes the last BN in each residual branch
    :type zero_init_residual: bool
    :param groups: ``groups`` for Conv2d.
    :type groups: int
    :param width_per_group: base width for Conv2d.
    :type width_per_group: int
    :param replace_stride_with_dilation: indicates if we should replace the 2x2 stride with a dilated convolution instead.
    :type replace_stride_with_dilation: list[bool] or None
    :param norm_layer: normalization module class or callable object.
    :type norm_layer: typing.Callable or nn.Module or None
    """
    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            layers: List[int],
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


@register_model_func
def resnet(
        depth: int,
        num_classes: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    """
    Instantiates a ResNet model for CIFAR datasets.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param depth: depth.
    :type depth: int
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet model.
    :rtype: ResNet4Cifar
    """
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202'
    n = (depth - 2) // 6
    model = ResNet4Cifar(BasicBlock, [n, n, n], num_classes, **kwargs)
    model_key = 'cifar{}-resnet{}'.format(num_classes, depth)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)
        model.load_state_dict(state_dict)
    elif pretrained:
        logger.warning(f'`pretrained` = True, but pretrained {model_key} model is not available')
    return model


@register_model_func
def resnet20(num_classes=10, pretrained=False, progress=True, **kwargs: Any):
    """
    ResNet-20 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-20 model.
    :rtype: ResNet4Cifar
    """
    return resnet(20, num_classes, pretrained, progress, **kwargs)


@register_model_func
def resnet32(num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> ResNet4Cifar:
    """
    ResNet-32 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-32 model.
    :rtype: ResNet4Cifar
    """
    return resnet(32, num_classes, pretrained, progress, **kwargs)


@register_model_func
def resnet44(num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> ResNet4Cifar:
    """
    ResNet-44 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-44 model.
    :rtype: ResNet4Cifar
    """
    return resnet(44, num_classes, pretrained, progress, **kwargs)


@register_model_func
def resnet56(num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> ResNet4Cifar:
    """
    ResNet-56 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-56 model.
    :rtype: ResNet4Cifar
    """
    return resnet(56, num_classes, pretrained, progress, **kwargs)


@register_model_func
def resnet110(num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> ResNet4Cifar:
    """
    ResNet-110 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-110 model.
    :rtype: ResNet4Cifar
    """
    return resnet(110, num_classes, pretrained, progress, **kwargs)


@register_model_func
def resnet1202(num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> ResNet4Cifar:
    """
    ResNet-1202 model.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param num_classes: 10 or 100 for CIFAR-10 or CIFAR-100, respectively.
    :type num_classes: int
    :param pretrained: if True, returns a model pre-trained on CIFAR dataset.
    :type pretrained: bool
    :param progress: if True, displays a progress bar of the download to stderr.
    :type progress: bool
    :return: ResNet-1202 model.
    :rtype: ResNet4Cifar
    """
    return resnet(1202, num_classes, pretrained, progress, **kwargs)

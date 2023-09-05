from collections import OrderedDict

from torch import nn

from .registry import register_adaptation_module
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_adaptation_module
class ConvReg(nn.Sequential):
    """
    `A convolutional regression for FitNets used in "Contrastive Representation Distillation" (CRD) <https://github.com/HobbitLong/RepDistiller/blob/34557d27282c83d49cff08b594944cf9570512bb/models/util.py#L131-L154>`_

    :param num_input_channels: ``in_channels`` for Conv2d.
    :type num_input_channels: int
    :param num_output_channels: ``out_channels`` for Conv2d.
    :type num_output_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: (int, int) or int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_relu: if True, uses ReLU as the last module.
    :type uses_relu: bool
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_relu=True):
        module_dict = OrderedDict()
        module_dict['conv'] =\
            nn.Conv2d(num_input_channels, num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        module_dict['bn'] = nn.BatchNorm2d(num_output_channels)
        if uses_relu:
            module_dict['relu'] = nn.ReLU(inplace=True)
        super().__init__(module_dict)

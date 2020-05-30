from collections import OrderedDict

from torch import nn

from common.constant import def_logger

logger = def_logger.getChild(__name__)
ADAPTATION_CLASS_DICT = dict()
MODULE_CLASS_DICT = nn.__dict__


def register_adaptation_module(cls):
    ADAPTATION_CLASS_DICT[cls.__name__] = cls
    return cls


@register_adaptation_module
class ConvReg(nn.Sequential):
    """
    Convolutional regression for FitNets used in "Contrastive Representation Distillation" (CRD)
    https://github.com/HobbitLong/RepDistiller/blob/34557d27282c83d49cff08b594944cf9570512bb/models/util.py#L131-L154
    But, hyperparameters are different from the original module due to larger input images in the target datasets
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_relu=True):
        module_dict = OrderedDict()
        module_dict['conv'] =\
            nn.Conv2d(num_input_channels, num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        module_dict['bn'] = nn.BatchNorm2d(num_output_channels)
        if uses_relu:
            module_dict['relu'] = nn.ReLU(inplace=True)
        super().__init__(module_dict)


def get_adaptation_module(class_name, *args, **kwargs):
    if class_name not in ADAPTATION_CLASS_DICT and class_name not in MODULE_CLASS_DICT:
        logger.info('No adaptation module called `{}` is registered.'.format(class_name))
        return None

    if class_name in ADAPTATION_CLASS_DICT:
        return ADAPTATION_CLASS_DICT[class_name](*args, **kwargs)
    return MODULE_CLASS_DICT[class_name](*args, **kwargs)

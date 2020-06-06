import os

import numpy as np
import torch
from torch import nn
from torch.jit.annotations import Tuple, List

from kdkit.common import main_util
from kdkit.common.constant import def_logger
from kdkit.models.official import get_vision_model
from kdkit.models.util import redesign_model
from myutils.common import file_util

logger = def_logger.getChild(__name__)
SPECIAL_CLASS_DICT = dict()


def register_special_module(cls):
    SPECIAL_CLASS_DICT[cls.__name__] = cls
    return cls


class SpecialModule(nn.Module):
    def __init__(self):
        super().__init__()

    def post_forward(self, *args, **kwargs):
        pass

    def post_process(self, *args, **kwargs):
        pass


@register_special_module
class EmptyModule(SpecialModule):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args


class Paraphraser4FactorTransfer(nn.Module):
    """
    Paraphraser for tactor transfer described in the supplementary material of
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    @staticmethod
    def make_tail_modules(num_output_channels, uses_bn):
        leaky_relu = nn.LeakyReLU(0.1)
        if uses_bn:
            return [nn.BatchNorm2d(num_output_channels), leaky_relu]
        return leaky_relu

    @classmethod
    def make_enc_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [
            nn.Conv2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding),
            *cls.make_tail_modules(num_output_channels, uses_bn)
        ]

    @classmethod
    def make_dec_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [
            nn.ConvTranspose2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding),
            *cls.make_tail_modules(num_output_channels, uses_bn)
        ]

    def __init__(self, k, num_input_channels, kernel_size=3, stride=1, padding=1, uses_bn=True):
        super().__init__()
        self.paraphrase_rate = k
        num_enc_output_channels = int(num_input_channels * k)
        self.encoder = nn.Sequential(
            *self.make_enc_modules(num_input_channels, num_input_channels,
                                   kernel_size, stride, padding, uses_bn),
            *self.make_enc_modules(num_input_channels, num_enc_output_channels,
                                   kernel_size, stride, padding, uses_bn),
            *self.make_enc_modules(num_enc_output_channels, num_enc_output_channels,
                                   kernel_size, stride, padding, uses_bn)
        )
        self.decoder = nn.Sequential(
            *self.make_dec_modules(num_enc_output_channels, num_enc_output_channels,
                                   kernel_size, stride, padding, uses_bn),
            *self.make_dec_modules(num_enc_output_channels, num_input_channels,
                                   kernel_size, stride, padding, uses_bn),
            *self.make_dec_modules(num_input_channels, num_input_channels,
                                   kernel_size, stride, padding, uses_bn)
        )

    def forward(self, z):
        if self.training:
            return self.decoder(self.encoder(z))
        return self.encoder(z)


class Translator4FactorTransfer(nn.Sequential):
    """
    Translator for factor transfer described in the supplementary material of
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    Note that "the student translator has the same three convolution layers as the paraphraser"
    """
    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, uses_bn=True):
        super().__init__(
            *Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_input_channels,
                                                         kernel_size, stride, padding, uses_bn),
            *Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_output_channels,
                                                         kernel_size, stride, padding, uses_bn),
            *Paraphraser4FactorTransfer.make_enc_modules(num_output_channels, num_output_channels,
                                                         kernel_size, stride, padding, uses_bn)
        )


@register_special_module
class Teacher4FactorTransfer(SpecialModule):
    """
    Teacher for factor transfer proposed in "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, base_model_config, input_module_path, paraphraser_params_config, paraphraser_ckpt, **kwargs):
        super().__init__()
        self.teacher_model = get_vision_model(base_model_config)
        self.input_module_path = input_module_path
        self.paraphraser = Paraphraser4FactorTransfer(**paraphraser_params_config)
        self.ckpt_file_path = paraphraser_ckpt
        if os.path.isfile(self.ckpt_file_path):
            self.paraphraser.load_state_dict(torch.load(self.ckpt_file_path, map_location='cpu'))

    def forward(self, x):
        return self.teacher_model(x)

    def post_forward(self, info_dict):
        self.paraphraser(info_dict[self.input_module_path]['output'])

    def post_process(self, *args, **kwargs):
        if main_util.is_main_process():
            file_util.make_parent_dirs(self.ckpt_file_path)
        main_util.save_on_master(self.paraphraser.state_dict(), self.ckpt_file_path)


@register_special_module
class Student4FactorTransfer(SpecialModule):
    """
    Student for factor transfer proposed in "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, student_model, input_module_path, translator_params_config, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.input_module_path = input_module_path
        self.translator = Translator4FactorTransfer(**translator_params_config)

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, info_dict):
        self.translator(info_dict[self.input_module_path]['output'])


@register_special_module
class Connector4DAB(SpecialModule):
    """
    Connector proposed in "Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"
    """

    @staticmethod
    def build_connector(conv_params_config, bn_params_config=None):
        module_list = [nn.Conv2d(**conv_params_config)]
        if bn_params_config is not None and len(bn_params_config) > 0:
            module_list.append(nn.BatchNorm2d(**bn_params_config))
        return nn.Sequential(*module_list)

    def __init__(self, student_model, connectors, **kwargs):
        super().__init__()
        self.student_model = student_model
        io_path_pairs = list()
        self.connector_dict = nn.ModuleDict()
        for connector_key, connector_params in connectors.items():
            self.connector_dict[connector_key] = \
                self.build_connector(connector_params['conv_params'], connector_params.get('bn_params', None))
            io_path_pairs.append((connector_key, connector_params['io'], connector_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, info_dict):
        for connector_key, io_type, module_path in self.io_path_pairs:
            self.connector_dict[connector_key](info_dict[module_path][io_type])


class Regressor4VID(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, eps, init_pred_var, **kwargs):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.soft_plus_param = \
            nn.Parameter(np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(out_channels))
        self.eps = eps
        self.init_pred_var = init_pred_var

    def forward(self, student_feature_map):
        pred_mean = self.regressor(student_feature_map)
        pred_var = torch.log(1.0 + torch.exp(self.soft_plus_param)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        return pred_mean, pred_var


@register_special_module
class VariationalDistributor4VID(SpecialModule):
    """
    "Variational Information Distillation for Knowledge Transfer"
    """

    def __init__(self, student_model, regressors, **kwargs):
        super().__init__()
        self.student_model = student_model
        io_path_pairs = list()
        self.regressor_dict = nn.ModuleDict()
        for regressor_key, regressor_params in regressors.items():
            self.regressor_dict[regressor_key] = Regressor4VID(**regressor_params)
            io_path_pairs.append((regressor_key, regressor_params['io'], regressor_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, info_dict):
        for regressor_key, io_type, module_path in self.io_path_pairs:
            self.regressor_dict[regressor_key](info_dict[module_path][io_type])


@register_special_module
class Linear4CCKD(SpecialModule):
    """
    Fully-connected layer to cope with a mismatch of feature representations of teacher and student network for
    "Correlation Congruence for Knowledge Distillation"
    """

    def __init__(self, input_module_path, linear_params, teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        self.model = teacher_model if teacher_model is not None else student_model
        self.input_module_path = input_module_path
        self.linear = nn.Linear(**linear_params)

    def forward(self, x):
        return self.model(x)

    def post_forward(self, info_dict):
        flat_outputs = torch.flatten(info_dict[self.input_module_path]['output'], 1)
        self.linear(flat_outputs)


class Normalizer4CRD(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


@register_special_module
class Linear4CRD(SpecialModule):
    """
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py
    """

    def __init__(self, input_module_path, linear_params, power=2,
                 teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        self.model = teacher_model if teacher_model is not None else student_model
        self.empty = nn.Sequential()
        self.input_module_path = input_module_path
        self.linear = nn.Linear(**linear_params)
        self.normalizer = Normalizer4CRD(power=power)

    def forward(self, x, supp_dict):
        # supp_dict is given to be hooked and stored in info_dict
        self.empty(supp_dict)
        return self.model(x)

    def post_forward(self, info_dict):
        flat_outputs = torch.flatten(info_dict[self.input_module_path]['output'], 1)
        z = self.linear(flat_outputs)
        self.normalizer(z)


@register_special_module
class HeadRCNN(SpecialModule):
    def __init__(self, head_rcnn, **kwargs):
        super().__init__()
        tmp_ref_model = kwargs.get('teacher_model', None)
        ref_model = kwargs.get('student_model', tmp_ref_model)
        if ref_model is None:
            raise ValueError('Either student_model or teacher_model has to be given.')

        self.transform = ref_model.transform
        self.seq = redesign_model(ref_model, head_rcnn, 'R-CNN')

    def forward(self, images, targets=None):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        return self.seq(images.tensors)


def get_special_module(class_name, *args, **kwargs):
    if class_name not in SPECIAL_CLASS_DICT:
        logger.info('No special module called `{}` is registered.'.format(class_name))
        return None

    instance = SPECIAL_CLASS_DICT[class_name](*args, **kwargs)
    return instance


def build_special_module(model_config, **kwargs):
    special_model_config = model_config.get('special', dict())
    special_model_type = special_model_config.get('type', None)
    if special_model_type is not None:
        special_model_params_config = special_model_config.get('params', None)
        if special_model_params_config is None:
            special_model_params_config = dict()
        return get_special_module(special_model_type, **kwargs, **special_model_params_config)
    return None

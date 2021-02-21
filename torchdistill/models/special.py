import os

import numpy as np
import torch
from torch import nn
from torch.jit.annotations import Tuple, List

from torchdistill.common.constant import def_logger
from torchdistill.models.util import wrap_if_distributed, load_module_ckpt, save_module_ckpt, redesign_model

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
    Paraphraser for factor transfer described in the supplementary material of
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    @staticmethod
    def make_tail_modules(num_output_channels, uses_bn):
        leaky_relu = nn.LeakyReLU(0.1)
        if uses_bn:
            return [nn.BatchNorm2d(num_output_channels), leaky_relu]
        return [leaky_relu]

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

    def __init__(self, teacher_model, minimal, input_module_path,
                 paraphraser_params, paraphraser_ckpt, uses_decoder, device, device_ids, distributed, **kwargs):
        super().__init__()
        if minimal is None:
            minimal = dict()
        
        special_teacher_model = build_special_module(minimal, teacher_model=teacher_model)
        model_type = 'original'
        teacher_ref_model = teacher_model
        if special_teacher_model is not None:
            teacher_ref_model = special_teacher_model
            model_type = type(teacher_ref_model).__name__

        self.teacher_model = redesign_model(teacher_ref_model, minimal, 'teacher', model_type)
        self.input_module_path = input_module_path
        self.paraphraser = \
            wrap_if_distributed(Paraphraser4FactorTransfer(**paraphraser_params), device, device_ids, distributed)
        self.ckpt_file_path = paraphraser_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(self.paraphraser, map_location, self.ckpt_file_path)
        self.uses_decoder = uses_decoder

    def forward(self, *args):
        with torch.no_grad():
            return self.teacher_model(*args)

    def post_forward(self, io_dict):
        if self.uses_decoder and not self.paraphraser.training:
            self.paraphraser.train()
        self.paraphraser(io_dict[self.input_module_path]['output'])

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.paraphraser, self.ckpt_file_path)


@register_special_module
class Student4FactorTransfer(SpecialModule):
    """
    Student for factor transfer proposed in "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """

    def __init__(self, student_model, input_module_path, translator_params, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.input_module_path = input_module_path
        self.translator = \
            wrap_if_distributed(Translator4FactorTransfer(**translator_params), device, device_ids, distributed)

    def forward(self, *args):
        return self.student_model(*args)

    def post_forward(self, io_dict):
        self.translator(io_dict[self.input_module_path]['output'])


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

    def __init__(self, student_model, connectors, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        io_path_pairs = list()
        self.connector_dict = nn.ModuleDict()
        for connector_key, connector_params in connectors.items():
            connector = self.build_connector(connector_params['conv_params'], connector_params.get('bn_params', None))
            self.connector_dict[connector_key] = wrap_if_distributed(connector, device, device_ids, distributed)
            io_path_pairs.append((connector_key, connector_params['io'], connector_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        for connector_key, io_type, module_path in self.io_path_pairs:
            self.connector_dict[connector_key](io_dict[module_path][io_type])


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

    def __init__(self, student_model, regressors, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        io_path_pairs = list()
        self.regressor_dict = nn.ModuleDict()
        for regressor_key, regressor_params in regressors.items():
            regressor = Regressor4VID(**regressor_params)
            self.regressor_dict[regressor_key] = wrap_if_distributed(regressor, device, device_ids, distributed)
            io_path_pairs.append((regressor_key, regressor_params['io'], regressor_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        for regressor_key, io_type, module_path in self.io_path_pairs:
            self.regressor_dict[regressor_key](io_dict[module_path][io_type])


@register_special_module
class Linear4CCKD(SpecialModule):
    """
    Fully-connected layer to cope with a mismatch of feature representations of teacher and student network for
    "Correlation Congruence for Knowledge Distillation"
    """

    def __init__(self, input_module, linear_params, device, device_ids, distributed,
                 teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        self.linear = wrap_if_distributed(nn.Linear(**linear_params), device, device_ids, distributed)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.linear(flat_outputs)


class Normalizer4CRD(nn.Module):
    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


@register_special_module
class Linear4CRD(SpecialModule):
    """
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py
    """

    def __init__(self, input_module_path, linear_params, device, device_ids, distributed, power=2,
                 teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.empty = nn.Sequential()
        self.input_module_path = input_module_path
        linear = nn.Linear(**linear_params)
        self.normalizer = wrap_if_distributed(Normalizer4CRD(linear, power=power), device, device_ids, distributed)

    def forward(self, x, supp_dict):
        # supp_dict is given to be hooked and stored in io_dict
        self.empty(supp_dict)
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path]['output'], 1)
        self.normalizer(flat_outputs)


@register_special_module
class HeadRCNN(SpecialModule):
    def __init__(self, head_rcnn, **kwargs):
        super().__init__()
        tmp_ref_model = kwargs.get('teacher_model', None)
        ref_model = kwargs.get('student_model', tmp_ref_model)
        if ref_model is None:
            raise ValueError('Either student_model or teacher_model has to be given.')

        self.transform = ref_model.transform
        self.seq = redesign_model(ref_model, head_rcnn, 'R-CNN', 'HeadRCNN')

    def forward(self, images, targets=None):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        return self.seq(images.tensors)


@register_special_module
class SSWrapper4SSKD(SpecialModule):
    """
    Semi-supervision wrapper for "Knowledge Distillation Meets Self-Supervision"
    """

    def __init__(self, input_module, feat_dim, ss_module_ckpt, device, device_ids, distributed, freezes_ss_module=False,
                 teacher_model=None, student_model=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        ss_module = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.ckpt_file_path = ss_module_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(ss_module, map_location, self.ckpt_file_path)
        self.ss_module = ss_module if is_teacher and freezes_ss_module \
            else wrap_if_distributed(ss_module, device, device_ids, distributed)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.ss_module(flat_outputs)

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.ss_module, self.ckpt_file_path)


@register_special_module
class VarianceBranch4PAD(SpecialModule):
    """
    Variance branch wrapper for "Prime-Aware Adaptive Distillation"
    """

    def __init__(self, student_model, input_module, feat_dim, var_estimator_ckpt,
                 device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        var_estimator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )
        self.ckpt_file_path = var_estimator_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(var_estimator, map_location, self.ckpt_file_path)
        self.var_estimator = wrap_if_distributed(var_estimator, device, device_ids, distributed)

    def forward(self, x):
        return self.student_model(x)

    def post_forward(self, io_dict):
        embed_outputs = io_dict[self.input_module_path][self.input_module_io].flatten(1)
        self.var_estimator(embed_outputs)

    def post_process(self, *args, **kwargs):
        save_module_ckpt(self.var_estimator, self.ckpt_file_path)


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

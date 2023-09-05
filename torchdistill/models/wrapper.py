import os

import numpy as np
import torch
from torch import nn
from torch.jit.annotations import Tuple, List
from torch.nn import functional

from .registry import register_auxiliary_model_wrapper, get_auxiliary_model_wrapper
from .util import wrap_if_distributed, load_module_ckpt, save_module_ckpt, redesign_model
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


class AuxiliaryModelWrapper(nn.Module):
    """
    An abstract auxiliary model wrapper.

    :meth:`forward`, :meth:`secondary_forward`, and :meth:`post_epoch_process` should be overridden by all subclasses.
    """
    def __init__(self):
        super().__init__()

    def secondary_forward(self, *args, **kwargs):
        pass

    def post_epoch_process(self, *args, **kwargs):
        pass


@register_auxiliary_model_wrapper
class EmptyModule(AuxiliaryModelWrapper):
    """
    An empty auxiliary model wrapper. This module returns input as output and is useful when you want to replace
    your teacher/student model with an empty model for saving inference time.
    e.g., Multi-stage knowledge distillation may have some stages that do not require either teacher or student models.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args


class Paraphraser4FactorTransfer(nn.Module):
    """
    Paraphraser for factor transfer (FT). This module is used at the 1st and 2nd stages of FT method.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param k: paraphrase rate.
    :type k: float
    :param num_input_channels: number of input channels.
    :type num_input_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_bn: if True, uses BatchNorm2d.
    :type uses_bn: bool
    :param uses_decoder: if True, uses decoder in :meth:`forward`.
    :type uses_decoder: bool
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

    def __init__(self, k, num_input_channels, kernel_size=3, stride=1, padding=1, uses_bn=True, uses_decoder=True):
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
        self.uses_decoder = uses_decoder

    def forward(self, z):
        if self.uses_decoder:
            return self.decoder(self.encoder(z))
        return self.encoder(z)


class Translator4FactorTransfer(nn.Sequential):
    """
    Translator for factor transfer (FT). This module is used at the 2nd stage of FT method.
    Note that "the student translator has the same three convolution layers as the paraphraser".

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param num_input_channels: number of input channels.
    :type num_input_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_bn: if True, uses BatchNorm2d.
    :type uses_bn: bool
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


@register_auxiliary_model_wrapper
class Teacher4FactorTransfer(AuxiliaryModelWrapper):
    """
    An auxiliary teacher model wrapper for factor transfer (FT), including paraphraser :class:`Paraphraser4FactorTransfer`.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param teacher_model: teacher model.
    :type teacher_model: nn.Module
    :param minimal: ``model_config`` for :meth:`build_auxiliary_model_wrapper` if you want to.
    :type minimal: dict or None
    :param input_module_path: path of module whose output is used as input to paraphraser.
    :type input_module_path: str
    :param paraphraser_kwargs: kwargs to instantiate :class:`Paraphraser4FactorTransfer`.
    :type paraphraser_kwargs: dict
    :param uses_decoder: ``uses_decoder`` for :class:`Paraphraser4FactorTransfer`.
    :type uses_decoder: bool
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, teacher_model, minimal, input_module_path,
                 paraphraser_kwargs, paraphraser_ckpt, uses_decoder, device, device_ids, distributed,
                 find_unused_parameters=None, **kwargs):
        super().__init__()
        if minimal is None:
            minimal = dict()
        
        auxiliary_teacher_model_wrapper = build_auxiliary_model_wrapper(minimal, teacher_model=teacher_model)
        model_type = 'original'
        teacher_ref_model = teacher_model
        if auxiliary_teacher_model_wrapper is not None:
            teacher_ref_model = auxiliary_teacher_model_wrapper
            model_type = type(teacher_ref_model).__name__

        self.teacher_model = redesign_model(teacher_ref_model, minimal, 'teacher', model_type)
        self.input_module_path = input_module_path
        paraphraser = Paraphraser4FactorTransfer(uses_decoder=uses_decoder, **paraphraser_kwargs)
        self.paraphraser = wrap_if_distributed(paraphraser, device, device_ids, distributed,
                                               find_unused_parameters=find_unused_parameters)
        self.ckpt_file_path = paraphraser_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(self.paraphraser, map_location, self.ckpt_file_path)
        self.uses_decoder = uses_decoder

    def forward(self, *args):
        with torch.no_grad():
            return self.teacher_model(*args)

    def secondary_forward(self, io_dict):
        if self.uses_decoder and not self.paraphraser.training:
            self.paraphraser.train()
        self.paraphraser(io_dict[self.input_module_path]['output'])

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.paraphraser, self.ckpt_file_path)


@register_auxiliary_model_wrapper
class Student4FactorTransfer(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for factor transfer (FT), including translator :class:`Translator4FactorTransfer`.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module_path: path of module whose output is used as input to paraphraser.
    :type input_module_path: str
    :param translator_kwargs: kwargs to instantiate :class:`Translator4FactorTransfer`.
    :type translator_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, student_model, input_module_path, translator_kwargs, device, device_ids, distributed,
                 find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed,
                                                 find_unused_parameters=find_unused_parameters)
        self.input_module_path = input_module_path
        self.translator = \
            wrap_if_distributed(Translator4FactorTransfer(**translator_kwargs), device, device_ids, distributed,
                                find_unused_parameters=find_unused_parameters)

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        self.translator(io_dict[self.input_module_path]['output'])


@register_auxiliary_model_wrapper
class Connector4DAB(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper with connector for distillation of activation boundaries (DAB).

    Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi: `"Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons" <https://ojs.aaai.org/index.php/AAAI/article/view/4264>`_ @ AAAI 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param connectors: connector keys and configurations.
    :type connectors: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    @staticmethod
    def build_connector(conv2d_kwargs, bn2d_kwargs=None):
        module_list = [nn.Conv2d(**conv2d_kwargs)]
        if bn2d_kwargs is not None and len(bn2d_kwargs) > 0:
            module_list.append(nn.BatchNorm2d(**bn2d_kwargs))
        return nn.Sequential(*module_list)

    def __init__(self, student_model, connectors, device, device_ids, distributed, find_unused_parameters=None,
                 **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        io_path_pairs = list()
        self.connector_dict = nn.ModuleDict()
        for connector_key, connector_config in connectors.items():
            connector = \
                self.build_connector(connector_config['conv2d_kwargs'], connector_config.get('bn2d_kwargs', None))
            self.connector_dict[connector_key] = \
                wrap_if_distributed(connector, device, device_ids, distributed, find_unused_parameters)
            io_path_pairs.append((connector_key, connector_config['io'], connector_config['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        for connector_key, io_type, module_path in self.io_path_pairs:
            self.connector_dict[connector_key](io_dict[module_path][io_type])


class Regressor4VID(nn.Module):
    """
    An auxiliary module for variational information distillation (VID).

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param mid_channels: number of output/input channels for the first/second convolution layer.
    :type mid_channels: int
    :param out_channels: number of output channels for the third convolution layer.
    :type out_channels: int
    :param eps: eps.
    :type eps: float
    :param init_pred_var: minimum variance introduced for numerical stability.
    :type init_pred_var: float
    """
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


@register_auxiliary_model_wrapper
class VariationalDistributor4VID(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for variational information distillation (VID), including translator :class:`Regressor4VID`.

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param regressors: regressor keys and configurations.
    :type regressors: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, student_model, regressors, device, device_ids, distributed, find_unused_parameters=None,
                 **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        io_path_pairs = list()
        self.regressor_dict = nn.ModuleDict()
        for regressor_key, regressor_config in regressors.items():
            regressor = Regressor4VID(**regressor_config['kwargs'])
            self.regressor_dict[regressor_key] = \
                wrap_if_distributed(regressor, device, device_ids, distributed, find_unused_parameters)
            io_path_pairs.append((regressor_key, regressor_config['io'], regressor_config['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        for regressor_key, io_type, module_path in self.io_path_pairs:
            self.regressor_dict[regressor_key](io_dict[module_path][io_type])


@register_auxiliary_model_wrapper
class Linear4CCKD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for correlation congruence for knowledge distillation (CCKD).
    Fully-connected layers cope with a mismatch of feature representations of teacher and student models.

    Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, Zhaoning Zhang: `"Correlation Congruence for Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV 2019 (2019)

    :param input_module: input module configuration.
    :type input_module: dict
    :param linear_kwargs: kwargs for Linear.
    :type linear_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, input_module, linear_kwargs, device, device_ids, distributed,
                 teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        self.linear = \
            wrap_if_distributed(nn.Linear(**linear_kwargs), device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.linear(flat_outputs)


class Normalizer4CRD(nn.Module):
    """
    An auxiliary module for contrastive representation distillation (CRD).

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param linear: linear module.
    :type linear: nn.Module
    :param power: the exponents.
    :type power: int
    """
    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


@register_auxiliary_model_wrapper
class Linear4CRD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for contrastive representation distillation (CRD), including translator :class:`Normalizer4CRD`.
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param input_module_path: path of module whose output will be flattened and then used as input to normalizer.
    :type input_module_path: str
    :param linear_kwargs: kwargs for Linear.
    :type linear_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param power: ``power`` for :class:`Normalizer4CRD`.
    :type power: int
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, input_module_path, linear_kwargs, device, device_ids, distributed, power=2,
                 teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.empty = nn.Sequential()
        self.input_module_path = input_module_path
        linear = nn.Linear(**linear_kwargs)
        self.normalizer = wrap_if_distributed(Normalizer4CRD(linear, power=power), device, device_ids, distributed,
                                              find_unused_parameters)

    def forward(self, x, supp_dict):
        # supp_dict is given to be hooked and stored in io_dict
        self.empty(supp_dict)
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path]['output'], 1)
        self.normalizer(flat_outputs)


@register_auxiliary_model_wrapper
class HeadRCNN(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for head network distillation (HND) and generalized head network distillation (GHND).

    * Yoshitomo Matsubara, Sabur Baidya, Davide Callegaro, Marco Levorato, Sameer Singh: `"Distilled Split Deep Neural Networks for Edge-Assisted Real-Time Systems" <https://dl.acm.org/doi/10.1145/3349614.3356022>`_ @ MobiCom 2019 Workshop on Hot Topics in Video Analytics and Intelligent Edges (2019)
    * Yoshitomo Matsubara, Marco Levorato: `"Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"  <https://arxiv.org/abs/2007.15818>`_ @ ICPR 2020 (2021)

    :param head_rcnn: head R-CNN configuration as ``model_config`` in :meth:`torchdistill.models.util.redesign_model`.
    :type head_rcnn: dict
    :param kwargs: ``teacher_model`` or ``student_model`` keys must be included. If both ``teacher_model`` and ``student_model`` are provided, ``student_model`` will be prioritized.
    :type kwargs: dict
    """
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


@register_auxiliary_model_wrapper
class SSWrapper4SSKD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for self-supervision knowledge distillation (SSKD).
    If both ``teacher_model`` and ``student_model`` are provided, ``student_model`` will be prioritized

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param input_module: input module configuration.
    :type input_module: dict
    :param feat_dim: number of input/output features for self-supervision module.
    :type feat_dim: int
    :param ss_module_ckpt: self-supervision module checkpoint file path.
    :type ss_module_ckpt: str
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param freezes_ss_module: if True, freezes self-supervision module.
    :type freezes_ss_module: bool
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, input_module, feat_dim, ss_module_ckpt, device, device_ids, distributed, freezes_ss_module=False,
                 teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)

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
            else wrap_if_distributed(ss_module, device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.ss_module(flat_outputs)

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.ss_module, self.ckpt_file_path)


@register_auxiliary_model_wrapper
class VarianceBranch4PAD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for prime-aware adaptive distillation (PAD).

    Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang, Yichen Wei: `"Prime-Aware Adaptive Distillation" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module: input module configuration.
    :type input_module: dict
    :param feat_dim: number of input/output features for self-supervision module.
    :type feat_dim: int
    :param var_estimator_ckpt: variance estimator module checkpoint file path.
    :type var_estimator_ckpt: str
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, student_model, input_module, feat_dim, var_estimator_ckpt,
                 device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
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
        self.var_estimator = wrap_if_distributed(var_estimator, device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        embed_outputs = io_dict[self.input_module_path][self.input_module_io].flatten(1)
        self.var_estimator(embed_outputs)

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.var_estimator, self.ckpt_file_path)


class AttentionBasedFusion(nn.Module):
    """
    An auxiliary module for knowledge review (KR). Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param mid_channels: number of output/input channels for the first/second convolution layer.
    :type mid_channels: int
    :param out_channels: number of output channels for the third convolution layer.
    :type out_channels: int
    """
    def __init__(self, in_channels, mid_channels, out_channels, uses_attention):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention_conv = None if not uses_attention \
            else nn.Sequential(nn.Conv2d(mid_channels * 2, 2, kernel_size=1), nn.Sigmoid())
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None, size=None):
        x = self.conv1(x)
        if self.attention_conv is not None:
            n, _, h, w = x.shape
            # upsample residual features
            y = functional.interpolate(y, (size, size), mode='nearest')
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.attention_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))

        y = self.conv2(x)
        return y, x


@register_auxiliary_model_wrapper
class Student4KnowledgeReview(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for knowledge review (KR). Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param student_model: student model.
    :type student_model: nn.Module
    :param abfs: attention based fusion configurations.
    :type abfs: list[dict]
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, student_model, abfs, device, device_ids, distributed, sizes=None,
                 find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        if sizes is None:
            sizes = [1, 7, 14, 28, 56]

        self.sizes = sizes
        abf_list = nn.ModuleList()
        num_abfs = len(abfs)
        io_path_pairs = list()
        for idx, abf_config in enumerate(abfs):
            abf = wrap_if_distributed(AttentionBasedFusion(uses_attention=idx < num_abfs - 1, **abf_config['kwargs']),
                                      device, device_ids, distributed, find_unused_parameters)
            abf_list.append(abf)
            io_path_pairs.append((abf_config['io'], abf_config['path']))

        self.abf_modules = abf_list[::-1]
        self.io_path_pairs = io_path_pairs[::-1]

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        feature_maps = [io_dict[module_path][io_type] for io_type, module_path in self.io_path_pairs]
        out_features, res_features = self.abf_modules[0](feature_maps[0])
        if len(self.sizes) > 1:
            for features, abf, size in zip(feature_maps[1:], self.abf_modules[1:], self.sizes[1:]):
                out_features, res_features = abf(features, res_features, size)


@register_auxiliary_model_wrapper
class Student4KTAAD(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for knowledge translation and adaptation + affinity distillation (KTAAD).
    Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module_path: path of module whose output is used as input to feature adapter and affinity adapter.
    :type input_module_path: str
    :param feature_adapter_config: feature adapter configuration.
    :type feature_adapter_config: dict
    :param affinity_adapter_config: affinity adapter configuration.
    :type affinity_adapter_config: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """
    def __init__(self, student_model, input_module_path, feature_adapter_config, affinity_adapter_config,
                 device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.input_module_path = input_module_path
        feature_adapter = nn.Sequential(
            nn.Conv2d(**feature_adapter_config['conv_kwargs']),
            nn.BatchNorm2d(**feature_adapter_config['bn_kwargs']), nn.ReLU(**feature_adapter_config['relu_kwargs'])
        )
        affinity_adapter = nn.Sequential(
            nn.Conv2d(**affinity_adapter_config['conv_kwargs'])
        )
        self.feature_adapter = \
            wrap_if_distributed(feature_adapter, device, device_ids, distributed, find_unused_parameters)
        self.affinity_adapter = \
            wrap_if_distributed(affinity_adapter, device, device_ids, distributed, find_unused_parameters)

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        feature_maps = io_dict[self.input_module_path]['output']
        self.feature_adapter(feature_maps)
        self.affinity_adapter(feature_maps)


def build_auxiliary_model_wrapper(model_config, **kwargs):
    """
    Builds an auxiliary model wrapper for either teacher or student models.

    :param model_config: configuration to build the auxiliary model wrapper. Should contain either 'teacher_model' or `student_model'.
    :type model_config: dict
    :return: auxiliary model wrapper.
    :rtype: nn.Module
    """
    auxiliary_model_wrapper_config = model_config.get('auxiliary_model_wrapper', dict())
    auxiliary_model_wrapper_key = auxiliary_model_wrapper_config.get('key', None)
    if auxiliary_model_wrapper_key is None:
        return None

    auxiliary_model_wrapper_kwargs = auxiliary_model_wrapper_config.get('kwargs', None)
    if auxiliary_model_wrapper_kwargs is None:
        auxiliary_model_wrapper_kwargs = dict()
    return get_auxiliary_model_wrapper(auxiliary_model_wrapper_key, **kwargs, **auxiliary_model_wrapper_kwargs)

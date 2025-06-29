import math

import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity

from .registry import register_loss_wrapper, register_mid_level_loss
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


def _extract_feature_map(io_dict, feature_map_config):
    io_type = feature_map_config['io']
    module_path = feature_map_config['path']
    return io_dict[module_path][io_type]


@register_loss_wrapper
class SimpleLossWrapper(nn.Module):
    """
    A simple loss wrapper module designed to use low-level loss modules (e.g., loss modules in PyTorch)
    in torchdistill's pipelines.

    :param low_level_loss: low-level loss module e.g., torch.nn.CrossEntropyLoss.
    :type low_level_loss: nn.Module
    :param kwargs: kwargs to configure what the wrapper passes ``low_level_loss``.
    :type kwargs: dict or None

    .. code-block:: YAML
       :caption: An example YAML to instantiate :class:`SimpleLossWrapper`.

        criterion_wrapper:
          key: 'SimpleLossWrapper'
          kwargs:
            input:
              is_from_teacher: False
              module_path: '.'
              io: 'output'
            target:
              uses_label: True
    """
    def __init__(self, low_level_loss, **kwargs):
        super().__init__()
        self.low_level_loss = low_level_loss
        input_config = kwargs['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io']
        target_config = kwargs.get('target', dict())
        self.uses_label = target_config.get('uses_label', False)
        self.is_target_from_teacher = target_config.get('is_from_teacher', None)
        self.target_module_path = target_config.get('module_path', None)
        self.target_key = target_config.get('io', None)

    @staticmethod
    def extract_value(io_dict, path, key):
        return io_dict[path][key]

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict,
                                         self.input_module_path, self.input_key)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict,
                                              self.target_module_path, self.target_key)
        return self.low_level_loss(input_batch, target_batch, *args, **kwargs)

    def __str__(self):
        return self.low_level_loss.__str__()


@register_loss_wrapper
class DictLossWrapper(SimpleLossWrapper):
    """
    A dict-based wrapper module designed to use low-level loss modules (e.g., loss modules in PyTorch)
    in torchdistill's pipelines. This is a subclass of :class:`SimpleLossWrapper` and useful for models whose forward
    output is dict.

    :param low_level_loss: low-level loss module e.g., torch.nn.CrossEntropyLoss.
    :type low_level_loss: nn.Module
    :param weights: dict contains keys that match the model's output dict keys and corresponding loss weights.
    :type weights: dict
    :param kwargs: kwargs to configure what the wrapper passes ``low_level_loss``.
    :type kwargs: dict or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`DictLossWrapper` for deeplabv3_resnet50 in torchvision, whose default output is a dict of outputs from its main and auxiliary branches with keys 'out' and 'aux' respectively.

        criterion_wrapper:
          key: 'DictLossWrapper'
          kwargs:
            input:
              is_from_teacher: False
              module_path: '.'
              io: 'output'
            target:
              uses_label: True
            weights:
              out: 1.0
              aux: 0.5
    """
    def __init__(self, low_level_loss, weights, **kwargs):
        super().__init__(low_level_loss, **kwargs)
        self.weights = weights

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict,
                                         self.input_module_path, self.input_key)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict,
                                              self.target_module_path, self.target_key)
        loss = None
        for key, weight in self.weights.items():
            sub_loss = self.low_level_loss(input_batch[key], target_batch, *args, **kwargs)
            if loss is None:
                loss = weight * sub_loss
            else:
                loss += weight * sub_loss
        return loss

    def __str__(self):
        return str(self.weights) + ' * ' + self.low_level_loss.__str__()


@register_mid_level_loss
class KDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation (KD) loss module.

    .. math::

       L_{KD} = \\alpha \cdot L_{CE} + (1 - \\alpha) \cdot \\tau^2 \cdot L_{KL}

    Geoffrey Hinton, Oriol Vinyals, Jeff Dean: `"Distilling the Knowledge in a Neural Network" <https://arxiv.org/abs/1503.02531>`_ @ NIPS 2014 Deep Learning and Representation Learning Workshop (2014)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param alpha: balancing factor for :math:`L_{CE}`, cross-entropy.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`) for :math:`L_{KL}`, KL divergence between class-probability distributions softened by :math:`\\tau`.
    :type beta: float or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None
    """
    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_io_dict, teacher_io_dict, targets=None, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        soft_loss = super().forward(torch.log_softmax(student_logits / self.temperature, dim=1),
                                    torch.softmax(teacher_logits / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


@register_mid_level_loss
class FSPLoss(nn.Module):
    """
    A loss module for the flow of solution procedure (FSP) matrix.

    Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim: `"A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning" <https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017)

    :param fsp_pairs: configuration of teacher-student module pairs to compute the loss for the FSP matrix.
    :type fsp_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`FSPLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'FSPLoss'
          kwargs:
            fsp_pairs:
              pair1:
                teacher_first:
                  io: 'input'
                  path: 'layer1'
                teacher_second:
                  io: 'output'
                  path: 'layer1'
                student_first:
                  io: 'input'
                  path: 'layer1'
                student_second:
                  io: 'output'
                  path: 'layer1'
                weight: 1
              pair2:
                teacher_first:
                  io: 'input'
                  path: 'layer2.1'
                teacher_second:
                  io: 'output'
                  path: 'layer2'
                student_first:
                  io: 'input'
                  path: 'layer2.1'
                student_second:
                  io: 'output'
                  path: 'layer2'
                weight: 1
    """
    def __init__(self, fsp_pairs, **kwargs):
        super().__init__()
        self.fsp_pairs = fsp_pairs

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = adaptive_max_pool2d(first_feature_map, (target_h, target_w))

        if second_h > target_h or second_w > target_w:
            second_feature_map = adaptive_max_pool2d(second_feature_map, (target_h, target_w))

        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        fsp_loss = 0
        batch_size = None
        for pair_name, pair_config in self.fsp_pairs.items():
            student_first_feature_map = _extract_feature_map(student_io_dict, pair_config['student_first'])
            student_second_feature_map = _extract_feature_map(student_io_dict, pair_config['student_second'])
            student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
            teacher_first_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher_first'])
            teacher_second_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher_second'])
            teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
            factor = pair_config.get('weight', 1)
            fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
            if batch_size is None:
                batch_size = student_first_feature_map.shape[0]
        return fsp_loss / batch_size


@register_mid_level_loss
class ATLoss(nn.Module):
    """
    A loss module for attention transfer (AT). Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py

    Sergey Zagoruyko, Nikos Komodakis: `"Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer" <https://openreview.net/forum?id=Sks9_ajex>`_ @ ICLR 2017 (2017)

    :param at_pairs: configuration of teacher-student module pairs to compute the loss for attention transfer.
    :type at_pairs: dict
    :param mode: reference to follow 'paper' or 'code'.
    :type mode: dict

    .. warning::
        There is a discrepancy between Eq. (2) in the paper and `the authors' implementation <https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23>`_
        as pointed out in `a paper <https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3>`_ and `an issue at the repository <https://github.com/szagoruyko/attention-transfer/issues/34>`_.
        Use ``mode`` = 'paper' instead of 'code' if you want to follow the equations in the paper.

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`ATLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'ATLoss'
          kwargs:
            at_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'layer3'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'layer4'
                weight: 1
            mode: 'code'
    """
    def __init__(self, at_pairs, mode='code', **kwargs):
        super().__init__()
        self.at_pairs = at_pairs
        self.mode = mode
        if mode not in ('code', 'paper'):
            raise ValueError('mode `{}` is not expected'.format(mode))

    @staticmethod
    def attention_transfer_paper(feature_map):
        return normalize(feature_map.pow(2).sum(1).flatten(1))

    def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer_paper(student_feature_map)
        at_teacher = self.attention_transfer_paper(teacher_feature_map)
        return torch.norm(at_student - at_teacher, dim=1).sum()

    @staticmethod
    def attention_transfer(feature_map):
        return normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_at_loss(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer(student_feature_map)
        at_teacher = self.attention_transfer(teacher_feature_map)
        return (at_student - at_teacher).pow(2).mean()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        at_loss = 0
        batch_size = None
        for pair_name, pair_config in self.at_pairs.items():
            student_feature_map = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            if self.mode == 'paper':
                at_loss += factor * self.compute_at_loss_paper(student_feature_map, teacher_feature_map)
            else:
                at_loss += factor * self.compute_at_loss(student_feature_map, teacher_feature_map)
            if batch_size is None:
                batch_size = len(student_feature_map)
        return at_loss / batch_size if self.mode == 'paper' else at_loss


@register_mid_level_loss
class PKTLoss(nn.Module):
    """
    A loss module for probabilistic knowledge transfer (PKT). Refactored https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py

    Nikolaos Passalis, Anastasios Tefas: `"Learning Deep Representations with Probabilistic Knowledge Transfer" <https://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html>`_ @ ECCV 2018 (2018)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param eps: constant to avoid zero division.
    :type eps: float

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`PKTLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'PKTLoss'
          kwargs:
            student_module_path: 'fc'
            student_module_io: 'input'
            teacher_module_path: 'fc'
            teacher_module_io: 'input'
            eps: 0.0000001
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, eps=0.0000001):
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.eps = eps

    def cosine_similarity_loss(self, student_outputs, teacher_outputs):
        # Normalize each vector by its norm
        norm_s = torch.sqrt(torch.sum(student_outputs ** 2, dim=1, keepdim=True))
        student_outputs = student_outputs / (norm_s + self.eps)
        student_outputs[student_outputs != student_outputs] = 0

        norm_t = torch.sqrt(torch.sum(teacher_outputs ** 2, dim=1, keepdim=True))
        teacher_outputs = teacher_outputs / (norm_t + self.eps)
        teacher_outputs[teacher_outputs != teacher_outputs] = 0

        # Calculate the cosine similarity
        student_similarity = torch.mm(student_outputs, student_outputs.transpose(0, 1))
        teacher_similarity = torch.mm(teacher_outputs, teacher_outputs.transpose(0, 1))

        # Scale cosine similarity to 0..1
        student_similarity = (student_similarity + 1.0) / 2.0
        teacher_similarity = (teacher_similarity + 1.0) / 2.0

        # Transform them into probabilities
        student_similarity = student_similarity / torch.sum(student_similarity, dim=1, keepdim=True)
        teacher_similarity = teacher_similarity / torch.sum(teacher_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        return torch.mean(teacher_similarity *
                          torch.log((teacher_similarity + self.eps) / (student_similarity + self.eps)))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_penultimate_outputs = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_penultimate_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        return self.cosine_similarity_loss(student_penultimate_outputs, teacher_penultimate_outputs)


@register_mid_level_loss
class FTLoss(nn.Module):
    """
    A loss module for factor transfer (FT). This loss module is used at the 2nd stage of FT method.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param p: the order of norm.
    :type p: int
    :param reduction: loss reduction type.
    :type reduction: str
    :param paraphraser_path: teacher model's paraphrase module path.
    :type paraphraser_path: str
    :param translator_path: student model's translator module path.
    :type translator_path: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`FTLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using auxiliary modules :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4FactorTransfer`.

        criterion:
          key: 'FTLoss'
          kwargs:
            p: 1
            reduction: 'mean'
            paraphraser_path: 'paraphraser'
            translator_path: 'translator'
    """
    def __init__(self, p=1, reduction='mean', paraphraser_path='paraphraser',
                 translator_path='translator', **kwargs):
        super().__init__()
        self.norm_p = p
        self.paraphraser_path = paraphraser_path
        self.translator_path = translator_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        paraphraser_flat_outputs = teacher_io_dict[self.paraphraser_path]['output'].flatten(1)
        translator_flat_outputs = student_io_dict[self.translator_path]['output'].flatten(1)
        norm_paraphraser_flat_outputs = paraphraser_flat_outputs / paraphraser_flat_outputs.norm(dim=1).unsqueeze(1)
        norm_translator_flat_outputs = translator_flat_outputs / translator_flat_outputs.norm(dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_translator_flat_outputs, norm_paraphraser_flat_outputs,
                                         reduction=self.reduction)
        ft_loss = torch.norm(norm_translator_flat_outputs - norm_paraphraser_flat_outputs, self.norm_p, dim=1)
        return ft_loss.mean() if self.reduction == 'mean' else ft_loss.sum()


@register_mid_level_loss
class AltActTransferLoss(nn.Module):
    """
    A loss module for distillation of activation boundaries (DAB). Refactored https://github.com/bhheo/AB_distillation/blob/master/cifar10_AB_distillation.py

    Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi: `"Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons" <https://ojs.aaai.org/index.php/AAAI/article/view/4264>`_ @ AAAI 2019 (2019)

    :param feature_pairs: configuration of teacher-student module pairs to compute the loss for distillation of activation boundaries.
    :type feature_pairs: dict
    :param margin: margin.
    :type margin: float
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AltActTransferLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Connector4DAB`.

        criterion:
          key: 'AltActTransferLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer1'
                student:
                  io: 'output'
                  path: 'connector_dict.connector1'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer2'
                student:
                  io: 'output'
                  path: 'connector_dict.connector2'
                weight: 1
              pair3:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'connector_dict.connector3'
                weight: 1
              pair4:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'connector_dict.connector4'
                weight: 1
            margin: 1.0
            reduction: 'mean'
    """
    def __init__(self, feature_pairs, margin, reduction, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs
        self.margin = margin
        self.reduction = reduction

    @staticmethod
    def compute_alt_act_transfer_loss(source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        dab_loss = 0
        batch_size = None
        for pair_name, pair_config in self.feature_pairs.items():
            student_feature_map = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            dab_loss += \
                factor * self.compute_alt_act_transfer_loss(student_feature_map, teacher_feature_map, self.margin)
            if batch_size is None:
                batch_size = student_feature_map.shape[0]
        return dab_loss / batch_size if self.reduction == 'mean' else dab_loss


@register_mid_level_loss
class RKDLoss(nn.Module):
    """
    A loss module for relational knowledge distillation (RKD). Refactored https://github.com/lenscloth/RKD/blob/master/metric/loss.py

    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho: `"Relational Knowledge Distillation" <https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_output_path: student module path whose output is used in this loss module.
    :type student_output_path: str
    :param teacher_output_path: teacher module path whose output is used in this loss module.
    :type teacher_output_path: str
    :param dist_factor: weight on distance-based RKD loss.
    :type dist_factor: float
    :param angle_factor: weight on angle-based RKD loss.
    :type angle_factor: float
    :param reduction: ``reduction`` for SmoothL1Loss.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`RKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'RKDLoss'
          kwargs:
            teacher_output_path: 'layer4'
            student_output_path: 'layer4'
            dist_factor: 1.0
            angle_factor: 2.0
            reduction: 'mean'
    """
    def __init__(self, student_output_path, teacher_output_path, dist_factor, angle_factor, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.dist_factor = dist_factor
        self.angle_factor = angle_factor
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction=reduction)

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def compute_rkd_distance_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.dist_factor is None or self.dist_factor == 0:
            return 0

        with torch.no_grad():
            t_d = self.pdist(teacher_flat_outputs, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student_flat_outputs, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        return self.smooth_l1_loss(d, t_d)

    def compute_rkd_angle_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.angle_factor is None or self.angle_factor == 0:
            return 0

        with torch.no_grad():
            td = (teacher_flat_outputs.unsqueeze(0) - teacher_flat_outputs.unsqueeze(1))
            norm_td = normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student_flat_outputs.unsqueeze(0) - student_flat_outputs.unsqueeze(1))
        norm_sd = normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        return self.smooth_l1_loss(s_angle, t_angle)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_flat_outputs = teacher_io_dict[self.teacher_output_path]['output'].flatten(1)
        student_flat_outputs = student_io_dict[self.student_output_path]['output'].flatten(1)
        rkd_distance_loss = self.compute_rkd_distance_loss(teacher_flat_outputs, student_flat_outputs)
        rkd_angle_loss = self.compute_rkd_angle_loss(teacher_flat_outputs, student_flat_outputs)
        return self.dist_factor * rkd_distance_loss + self.angle_factor * rkd_angle_loss


@register_mid_level_loss
class VIDLoss(nn.Module):
    """
    A loss module for variational information distillation (VID). Referred to https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param feature_pairs: configuration of teacher-student module pairs to compute the loss for variational information distillation.
    :type feature_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`VIDLoss` for a teacher-student pair of ResNet-50 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.VariationalDistributor4VID` for the student model.

        criterion:
          key: 'VIDLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer1'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor1'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer2'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor2'
                weight: 1
              pair3:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor3'
                weight: 1
              pair4:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor4'
                weight: 1
            margin: 1.0
    """
    def __init__(self, feature_pairs, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        vid_loss = 0
        for pair_name, pair_config in self.feature_pairs.items():
            pred_mean, pred_var = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            neg_log_prob = 0.5 * ((pred_mean - teacher_feature_map) ** 2 / pred_var + torch.log(pred_var))
            vid_loss += factor * neg_log_prob.mean()
        return vid_loss


@register_mid_level_loss
class CCKDLoss(nn.Module):
    """
    A loss module for correlation congruence for knowledge distillation (CCKD).

    Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, Zhaoning Zhang: `"Correlation Congruence for Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV 2019 (2019)

    :param student_linear_path: student model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CCKD`.
    :type student_linear_path: str
    :param teacher_linear_path: teacher model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CCKD`.
    :type teacher_linear_path: str
    :param kernel_config: kernel ('gaussian' or 'bilinear') configuration.
    :type kernel_config: dict
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`CCKDLoss` for a teacher-student pair of ResNet-50 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Linear4CCKD` for the teacher and student models.

        criterion:
          key: 'CCKDLoss'
          kwargs:
            teacher_linear_path: 'linear'
            student_linear_path: 'linear'
            kernel_params:
              key: 'gaussian'
              gamma: 0.4
              max_p: 2
            reduction: 'batchmean'
    """
    def __init__(self, student_linear_path, teacher_linear_path, kernel_config, reduction, **kwargs):
        super().__init__()
        self.student_linear_path = student_linear_path
        self.teacher_linear_path = teacher_linear_path
        self.kernel_type = kernel_config['type']
        if self.kernel_type == 'gaussian':
            self.gamma = kernel_config['gamma']
            self.max_p = kernel_config['max_p']
        elif self.kernel_type not in ('bilinear', 'gaussian'):
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))
        self.reduction = reduction

    @staticmethod
    def compute_cc_mat_by_bilinear_pool(linear_outputs):
        return torch.matmul(linear_outputs, torch.t(linear_outputs))

    def compute_cc_mat_by_gaussian_rbf(self, linear_outputs):
        row_list = list()
        for index, linear_output in enumerate(linear_outputs):
            row = 1
            right_term = torch.matmul(linear_output, torch.t(linear_outputs))
            for p in range(1, self.max_p + 1):
                left_term = ((2 * self.gamma) ** p) / (math.factorial(p))
                row += left_term * (right_term ** p)

            row *= math.exp(-2 * self.gamma)
            row_list.append(row.squeeze(0))
        return torch.stack(row_list)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_path]['output']
        student_linear_outputs = student_io_dict[self.student_linear_path]['output']
        batch_size = teacher_linear_outputs.shape[0]
        if self.kernel_type == 'bilinear':
            teacher_cc = self.compute_cc_mat_by_bilinear_pool(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_bilinear_pool(student_linear_outputs)
        elif self.kernel_type == 'gaussian':
            teacher_cc = self.compute_cc_mat_by_gaussian_rbf(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_gaussian_rbf(student_linear_outputs)
        else:
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))

        cc_loss = torch.dist(student_cc, teacher_cc, 2)
        return cc_loss / (batch_size ** 2) if self.reduction == 'batchmean' else cc_loss


@register_mid_level_loss
class SPKDLoss(nn.Module):
    """
    A loss module for similarity-preserving knowledge distillation (SPKD).

    Frederick Tung, Greg Mori: `"Similarity-Preserving Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV2019 (2019)

    :param student_output_path: student module path whose output is used in this loss module.
    :type student_output_path: str
    :param teacher_output_path: teacher module path whose output is used in this loss module.
    :type teacher_output_path: str
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`SPKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'SPKDLoss'
          kwargs:
            teacher_output_path: 'layer4'
            student_output_path: 'layer4'
            reduction: 'batchmean'
    """
    def __init__(self, student_output_path, teacher_output_path, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss


@register_mid_level_loss
class CRDLoss(nn.Module):
    """
    A loss module for contrastive representation distillation (CRD). Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param student_norm_module_path: student model's normalizer module path (:class:`torchdistill.models.wrapper.Normalizer4CRD` in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`).
    :type student_norm_module_path: str
    :param student_empty_module_path: student model's empty module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`.
    :type student_empty_module_path: str
    :param teacher_norm_module_path: teacher model's normalizer module path (:class:`torchdistill.models.wrapper.Normalizer4CRD` in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`).
    :type teacher_norm_module_path: str
    :param input_size: number of input features.
    :type input_size: int
    :param output_size: number of output features.
    :type output_size: int
    :param num_negative_samples: number of negative samples.
    :type num_negative_samples: int
    :param num_samples: number of samples.
    :type num_samples: int
    :param temperature: temperature to adjust concentration level (not the temperature for :class:`KDLoss`).
    :type temperature: float
    :param momentum: momentum.
    :type momentum: float
    :param eps: eps.
    :type eps: float

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`CRDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Linear4CRD` for the teacher and student models.

        criterion:
          key: 'CRDLoss'
          kwargs:
            teacher_norm_module_path: 'normalizer'
            student_norm_module_path: 'normalizer'
            student_empty_module_path: 'empty'
            input_size: *feature_dim
            output_size: &num_samples 1281167
            num_negative_samples: *num_negative_samples
            num_samples: *num_samples
            temperature: 0.07
            momentum: 0.5
            eps: 0.0000001
    """

    def init_prob_alias(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())

        k = len(probs)
        self.probs = torch.zeros(k)
        self.alias = torch.zeros(k, dtype=torch.int64)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.probs[kk] = k * prob
            if self.probs[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.probs[large] = (self.probs[large] - 1.0) + self.probs[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.probs[last_one] = 1

    def __init__(self, student_norm_module_path, student_empty_module_path, teacher_norm_module_path,
                 input_size, output_size, num_negative_samples, num_samples, temperature=0.07, momentum=0.5, eps=1e-7):
        super().__init__()
        self.student_norm_module_path = student_norm_module_path
        self.student_empty_module_path = student_empty_module_path
        self.teacher_norm_module_path = teacher_norm_module_path
        self.eps = eps
        self.unigrams = torch.ones(output_size)
        self.num_negative_samples = num_negative_samples
        self.num_samples = num_samples
        self.register_buffer('params', torch.tensor([num_negative_samples, temperature, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(input_size / 3)
        self.register_buffer('memory_v1', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.probs, self.alias = None, None
        self.init_prob_alias(self.unigrams)

    def draw(self, n):
        # Draw n samples from multinomial
        k = self.alias.size(0)
        kk = torch.zeros(n, dtype=torch.long, device=self.prob.device).random_(0, k)
        prob = self.probs.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())
        return oq + oj

    def contrast_memory(self, student_embed, teacher_embed, pos_indices, contrast_idx=None):
        param_k = int(self.params[0].item())
        param_t = self.params[1].item()
        z_v1 = self.params[2].item()
        z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batch_size = student_embed.size(0)
        output_size = self.memory_v1.size(0)
        input_size = self.memory_v1.size(1)

        # original score computation
        if contrast_idx is None:
            contrast_idx = self.draw(batch_size * (self.num_negative_samples + 1)).view(batch_size, -1)
            contrast_idx.select(1, 0).copy_(pos_indices.data)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, contrast_idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batch_size, param_k + 1, input_size)
        out_v2 = torch.bmm(weight_v1, teacher_embed.view(batch_size, input_size, 1))
        out_v2 = torch.exp(torch.div(out_v2, param_t))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, contrast_idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batch_size, param_k + 1, input_size)
        out_v1 = torch.bmm(weight_v2, student_embed.view(batch_size, input_size, 1))
        out_v1 = torch.exp(torch.div(out_v1, param_t))

        # set z if haven't been set yet
        if z_v1 < 0:
            self.params[2] = out_v1.mean() * output_size
            z_v1 = self.params[2].clone().detach().item()
            logger.info('normalization constant z_v1 is set to {:.1f}'.format(z_v1))
        if z_v2 < 0:
            self.params[3] = out_v2.mean() * output_size
            z_v2 = self.params[3].clone().detach().item()
            logger.info('normalization constant z_v2 is set to {:.1f}'.format(z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, z_v1).contiguous()
        out_v2 = torch.div(out_v2, z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, pos_indices.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(student_embed, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, pos_indices, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, pos_indices.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(teacher_embed, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, pos_indices, updated_v2)
        return out_v1, out_v2

    def compute_contrast_loss(self, x):
        batch_size = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        pn = 1 / float(self.num_samples)

        # loss for positive pair
        p_pos = x.select(1, 0)
        log_d1 = torch.div(p_pos, p_pos.add(m * pn + self.eps)).log_()

        # loss for K negative pair
        p_neg = x.narrow(1, 1, m)
        log_d0 = torch.div(p_neg.clone().fill_(m * pn), p_neg.add(m * pn + self.eps)).log_()

        loss = - (log_d1.sum(0) + log_d0.view(-1, 1).sum(0)) / batch_size
        return loss

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        # pos_idx: the indices of these positive samples in the dataset, size [batch_size]
        # contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        teacher_linear_outputs = teacher_io_dict[self.teacher_norm_module_path]['output']
        student_linear_outputs = student_io_dict[self.student_norm_module_path]['output']
        supp_dict = student_io_dict[self.student_empty_module_path]['input']
        pos_idx, contrast_idx = supp_dict['pos_idx'], supp_dict.get('contrast_idx', None)
        device = student_linear_outputs.device
        pos_idx = pos_idx.to(device)
        if contrast_idx is not None:
            contrast_idx = contrast_idx.to(device)

        if device != self.probs.device:
            self.probs.to(device)
            self.alias.to(device)
            self.to(device)

        out_s, out_t = self.contrast_memory(student_linear_outputs, teacher_linear_outputs, pos_idx, contrast_idx)
        student_contrast_loss = self.compute_contrast_loss(out_s)
        teacher_contrast_loss = self.compute_contrast_loss(out_t)
        loss = student_contrast_loss + teacher_contrast_loss
        return loss


@register_mid_level_loss
class AuxSSKDLoss(nn.CrossEntropyLoss):
    """
    A loss module for self-supervision knowledge distillation (SSKD) that treats contrastive prediction as
    a self-supervision task (auxiliary task). This loss module is used at the 1st stage of SSKD method.
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param module_path: model's self-supervision module path.
    :type module_path: str
    :param module_io: 'input' or 'output' of the module in the model.
    :type module_io: str
    :param reduction: ``reduction`` for CrossEntropyLoss.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AuxSSKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.SSWrapper4SSKD` for teacher model.

        criterion:
          key: 'AuxSSKDLoss'
          kwargs:
            module_path: 'ss_module'
            module_io: 'output'
            reduction: 'mean'
    """
    def __init__(self, module_path='ss_module', module_io='output', reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.module_path = module_path
        self.module_io = module_io

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        ss_module_outputs = teacher_io_dict[self.module_path][self.module_io]
        device = ss_module_outputs.device
        batch_size = ss_module_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = (torch.arange(batch_size) % 4 == 0)
        aug_indices = (torch.arange(batch_size) % 4 != 0)
        normal_rep = ss_module_outputs[normal_indices]
        aug_rep = ss_module_outputs[aug_indices]
        normal_rep = normal_rep.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        cos_similarities = cosine_similarity(aug_rep, normal_rep, dim=1)
        targets = torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        targets = targets[:three_forth_batch_size].long().to(device)
        return super().forward(cos_similarities, targets)


@register_mid_level_loss
class SSKDLoss(nn.Module):
    """
    A loss module for self-supervision knowledge distillation (SSKD).
    This loss module is used at the 2nd stage of SSKD method. Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_linear_path: student model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SSWrapper4SSKD`.
    :type student_linear_path: str
    :param teacher_linear_path: teacher model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SSWrapper4SSKD`.
    :type teacher_linear_path: str
    :param student_ss_module_path: student model's self-supervision module path.
    :type student_ss_module_path: str
    :param teacher_ss_module_path: teacher model's self-supervision module path.
    :type teacher_ss_module_path: str
    :param kl_temp: temperature to soften teacher and student's class-probability distributions for KL divergence given original data.
    :type kl_temp: float
    :param ss_temp: temperature to soften teacher and student's self-supervision cosine similarities for KL divergence.
    :type ss_temp: float
    :param tf_temp: temperature to soften teacher and student's class-probability distributions for KL divergence given augmented data by transform.
    :type tf_temp: float
    :param ss_ratio: ratio of samples with the smallest error levels used for self-supervision.
    :type ss_ratio: float
    :param tf_ratio: ratio of samples with the smallest error levels used for transform.
    :type tf_ratio: float
    :param student_linear_module_io: 'input' or 'output' of the linear module in the student model.
    :type student_linear_module_io: str
    :param teacher_linear_module_io: 'input' or 'output' of the linear module in the teacher model.
    :type teacher_linear_module_io: str
    :param student_ss_module_io: 'input' or 'output' of the self-supervision module in the student model.
    :type student_ss_module_io: str
    :param teacher_ss_module_io: 'input' or 'output' of the self-supervision module in the teacher model.
    :type teacher_ss_module_io: str
    :param loss_weights: weights for 1) cross-entropy, 2) KL divergence for the original data, 3) KL divergence for self-supervision cosine similarities, and 4) KL divergence for the augmented data by transform.
    :type loss_weights: list[float] or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`SSKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.SSWrapper4SSKD` for the teacher and student models.

        criterion:
          key: 'SSKDLoss'
          kwargs:
            student_linear_module_path: 'model.fc'
            teacher_linear_module_path: 'model.fc'
            student_ss_module_path: 'ss_module'
            teacher_ss_module_path: 'ss_module'
            kl_temp: 4.0
            ss_temp: 0.5
            tf_temp: 4.0
            ss_ratio: 0.75
            tf_ratio: 1.0
            loss_weights: [1.0, 0.9, 10.0, 2.7]
            reduction: 'batchmean'
    """
    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_ss_module_path,
                 teacher_ss_module_path, kl_temp, ss_temp, tf_temp, ss_ratio, tf_ratio,
                 student_linear_module_io='output', teacher_linear_module_io='output',
                 student_ss_module_io='output', teacher_ss_module_io='output',
                 loss_weights=None, reduction='batchmean', **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        self.kl_temp = kl_temp
        self.ss_temp = ss_temp
        self.tf_temp = tf_temp
        self.ss_ratio = ss_ratio
        self.tf_ratio = tf_ratio
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_ss_module_path = student_ss_module_path
        self.student_ss_module_io = student_ss_module_io
        self.teacher_ss_module_path = teacher_ss_module_path
        self.teacher_ss_module_io = teacher_ss_module_io

    @staticmethod
    def compute_cosine_similarities(ss_module_outputs, normal_indices, aug_indices,
                                    three_forth_batch_size, one_forth_batch_size):
        normal_feat = ss_module_outputs[normal_indices]
        aug_feat = ss_module_outputs[aug_indices]
        normal_feat = normal_feat.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_feat = aug_feat.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        return cosine_similarity(aug_feat, normal_feat, dim=1)

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        device = student_linear_outputs.device
        batch_size = student_linear_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = (torch.arange(batch_size) % 4 == 0)
        aug_indices = (torch.arange(batch_size) % 4 != 0)
        ce_loss = self.cross_entropy_loss(student_linear_outputs[normal_indices], targets)
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs[normal_indices] / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs[normal_indices] / self.kl_temp, dim=1))
        kl_loss *= (self.kl_temp ** 2)

        # error level ranking
        aug_knowledges = torch.softmax(teacher_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        aug_targets = targets.unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(aug_knowledges, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.tf_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_tf = torch.sort(indices)[0]

        student_ss_module_outputs = student_io_dict[self.student_ss_module_path][self.student_ss_module_io]
        teacher_ss_module_outputs = teacher_io_dict[self.teacher_ss_module_path][self.teacher_ss_module_io]

        s_cos_similarities = self.compute_cosine_similarities(student_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = self.compute_cosine_similarities(teacher_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = t_cos_similarities.detach()

        aug_targets = \
            torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(t_cos_similarities, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.ss_ratio)
        indices = indices[:correct_num+wrong_keep]
        distill_index_ss = torch.sort(indices)[0]

        ss_loss = self.kldiv_loss(torch.log_softmax(s_cos_similarities[distill_index_ss] / self.ss_temp, dim=1),
                                  torch.softmax(t_cos_similarities[distill_index_ss] / self.ss_temp, dim=1))
        ss_loss *= (self.ss_temp ** 2)
        log_aug_outputs = torch.log_softmax(student_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        tf_loss = self.kldiv_loss(log_aug_outputs[distill_index_tf], aug_knowledges[distill_index_tf])
        tf_loss *= (self.tf_temp ** 2)
        total_loss = 0
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, ss_loss, tf_loss]):
            total_loss += loss_weight * loss
        return total_loss


@register_mid_level_loss
class PADL2Loss(nn.Module):
    """
    A loss module for prime-aware adaptive distillation (PAD) with L2 loss. This loss module is used at the 2nd stage of PAD method.

    Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang, Yichen Wei: `"Prime-Aware Adaptive Distillation" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_embed_module_path: student model's embedding module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.VarianceBranch4PAD`.
    :type student_embed_module_path: str
    :param teacher_embed_module_path: teacher model's embedding module path.
    :type teacher_embed_module_path: str
    :param student_embed_module_io: 'input' or 'output' of the embedding module in the student model.
    :type student_embed_module_io: str
    :param teacher_embed_module_io: 'input' or 'output' of the embedding module in the teacher model.
    :type teacher_embed_module_io: str
    :param module_path: student model's variance estimator module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.VarianceBranch4PAD`.
    :type module_path: str
    :param module_io: 'input' or 'output' of the variance estimator module in the student model.
    :type module_io: str
    :param eps: constant to avoid zero division.
    :type eps: float
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`PADL2Loss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.VarianceBranch4PAD` for the student model.

        criterion:
          key: 'PADL2Loss'
          kwargs:
            student_embed_module_path: 'student_model.avgpool'
            student_embed_module_io: 'output'
            teacher_embed_module_path: 'avgpool'
            teacher_embed_module_io: 'output'
            module_path: 'var_estimator'
            module_io: 'output'
            eps: 0.000001
            reduction: 'mean'
    """
    def __init__(self, student_embed_module_path, teacher_embed_module_path,
                 student_embed_module_io='output', teacher_embed_module_io='output',
                 module_path='var_estimator', module_io='output', eps=1e-6, reduction='mean', **kwargs):
        super().__init__()
        self.student_embed_module_path = student_embed_module_path
        self.teacher_embed_module_path = teacher_embed_module_path
        self.student_embed_module_io = student_embed_module_io
        self.teacher_embed_module_io = teacher_embed_module_io
        self.module_path = module_path
        self.module_io = module_io
        self.eps = eps
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        log_variances = student_io_dict[self.module_path][self.module_io]
        student_embed_outputs = student_io_dict[self.student_embed_module_path][self.student_embed_module_io].flatten(1)
        teacher_embed_outputs = teacher_io_dict[self.teacher_embed_module_path][self.teacher_embed_module_io].flatten(1)
        # The author's provided code takes average of losses
        squared_losses = torch.mean(
            (teacher_embed_outputs - student_embed_outputs) ** 2 / (self.eps + torch.exp(log_variances))
            + log_variances, dim=1
        )
        return squared_losses.mean() if self.reduction == 'mean' else squared_losses.sum()


@register_mid_level_loss
class HierarchicalContextLoss(nn.Module):
    """
    A loss module for knowledge review (KR) method. Referred to https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param student_module_path: student model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Student4KnowledgeReview`.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param reduction: ``reduction`` for MSELoss.
    :type reduction: str or None
    :param output_sizes: output sizes of adaptive_avg_pool2d.
    :type output_sizes: list[int] or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`HierarchicalContextLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Student4KnowledgeReview` for the student model.

        criterion:
          key: 'HierarchicalContextLoss'
          kwargs:
            student_module_path: 'abf_modules.4'
            student_module_io: 'output'
            teacher_module_path: 'layer1.-1.relu'
            teacher_module_io: 'input'
            reduction: 'mean'
            output_sizes: [4, 2, 1]
    """
    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 reduction='mean', output_sizes=None, **kwargs):
        super().__init__()
        if output_sizes is None:
            output_sizes = [4, 2, 1]

        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.criteria = nn.MSELoss(reduction=reduction)
        self.output_sizes = output_sizes

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features, _ = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_features = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        _, _, h, _ = student_features.shape
        loss = self.criteria(student_features, teacher_features)
        weight = 1.0
        total_weight = 1.0
        for k in self.output_sizes:
            if k >= h:
                continue

            proc_student_features = adaptive_avg_pool2d(student_features, (k, k))
            proc_teacher_features = adaptive_avg_pool2d(teacher_features, (k, k))
            weight /= 2.0
            loss += weight * self.criteria(proc_student_features, proc_teacher_features)
            total_weight += weight
        return loss / total_weight


@register_mid_level_loss
class RegularizationLoss(nn.Module):
    """
    A regularization loss module.

    :param module_path: module path.
    :type module_path: str
    :param module_io: 'input' or 'output' of the module in the student model.
    :type module_io: str
    :param is_from_teacher: True if you use teacher's I/O dict. Otherwise, you use student's I/O dict.
    :type is_from_teacher: bool
    :param p: the order of norm.
    :type p: int
    """
    def __init__(self, module_path, io_type='output', is_from_teacher=False, p=1, **kwargs):
        super().__init__()
        self.module_path = module_path
        self.io_type = io_type
        self.is_from_teacher = is_from_teacher
        self.norm_p = p

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        io_dict = teacher_io_dict if self.is_from_teacher else student_io_dict
        z = io_dict[self.module_path][self.io_type]
        return z.norm(p=self.norm_p)


@register_mid_level_loss
class KTALoss(nn.Module):
    """
    A loss module for knowledge translation and adaptation (KTA).
    This loss module is used at the 2nd stage of KTAAD method.

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param p: the order of norm for differences between normalized feature adapter's (flattened) output and knowledge translator's (flattened) output.
    :type p: int
    :param q: the order of norm for the denominator to normalize feature adapter (flattened) output.
    :type q: int
    :param reduction: loss reduction type.
    :type reduction: str
    :param knowledge_translator_path: knowledge translator module path.
    :type knowledge_translator_path: str
    :param feature_adapter_path: feature adapter module path.
    :type feature_adapter_path: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`KTALoss` for a teacher-student pair of DeepLabv3 with ResNet50 and LRASPP with MobileNet v3 (Large) in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4KTAAD` for the teacher and student models.

        criterion:
          key: 'KTALoss'
          kwargs:
            p: 1
            q: 2
            reduction: 'mean'
            knowledge_translator_path: 'paraphraser.encoder'
            feature_adapter_path: 'feature_adapter'
    """
    def __init__(self, p=1, q=2, reduction='mean', knowledge_translator_path='paraphraser',
                 feature_adapter_path='feature_adapter', **kwargs):
        super().__init__()
        self.norm_p = p
        self.norm_q = q
        self.knowledge_translator_path = knowledge_translator_path
        self.feature_adapter_path = feature_adapter_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        knowledge_translator_flat_outputs = teacher_io_dict[self.knowledge_translator_path]['output'].flatten(1)
        feature_adapter_flat_outputs = student_io_dict[self.feature_adapter_path]['output'].flatten(1)
        norm_knowledge_translator_flat_outputs = \
            knowledge_translator_flat_outputs / \
            knowledge_translator_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        norm_feature_adapter_flat_outputs = \
            feature_adapter_flat_outputs / feature_adapter_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_feature_adapter_flat_outputs, norm_knowledge_translator_flat_outputs,
                                         reduction=self.reduction)
        kta_loss = \
            torch.norm(norm_feature_adapter_flat_outputs - norm_knowledge_translator_flat_outputs, self.norm_p, dim=1)
        return kta_loss.mean() if self.reduction == 'mean' else kta_loss.sum()


@register_mid_level_loss
class AffinityLoss(nn.Module):
    """
    A loss module for affinity distillation in KTA. This loss module is used at the 2nd stage of KTAAD method.

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_module_path: student model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Student4KTAAD`.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Teacher4FactorTransfer`.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param reduction: loss reduction type.
    :type reduction: str or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AffinityLoss` for a teacher-student pair of DeepLabv3 with ResNet50 and LRASPP with MobileNet v3 (Large) in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4KTAAD` for the teacher and student models.

        criterion:
          key: 'AffinityLoss'
          kwargs:
            student_module_path: 'affinity_adapter'
            student_module_io: 'output'
            teacher_module_path: 'paraphraser.encoder'
            teacher_module_io: 'output'
            reduction: 'mean'
    """
    def __init__(self, student_module_path, teacher_module_path,
                 student_module_io='output', teacher_module_io='output', reduction='mean', **kwargs):
        super().__init__()
        self.student_module_path = student_module_path
        self.teacher_module_path = teacher_module_path
        self.student_module_io = student_module_io
        self.teacher_module_io = teacher_module_io
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_flat_outputs = student_io_dict[self.student_module_path][self.student_module_io].flatten(2)
        teacher_flat_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io].flatten(2)
        batch_size, ch_size, hw = student_flat_outputs.shape
        student_flat_outputs = student_flat_outputs / student_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        teacher_flat_outputs = teacher_flat_outputs / teacher_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        total_squared_losses = torch.zeros(batch_size).to(student_flat_outputs.device)
        for i in range(ch_size):
            total_squared_losses += (
                (torch.bmm(student_flat_outputs[:, i].unsqueeze(2), student_flat_outputs[:, i].unsqueeze(1))
                 - torch.bmm(teacher_flat_outputs[:, i].unsqueeze(2), teacher_flat_outputs[:, i].unsqueeze(1))) / hw
            ).norm(p=2, dim=(1, 2))
        return total_squared_losses.mean() if self.reduction == 'mean' else total_squared_losses.sum()


@register_mid_level_loss
class ChSimLoss(nn.Module):
    """
    A loss module for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    Refactored https://github.com/ADLab-AutoDrive/ICKD/blob/main/ImageNet/torchdistill/losses/single.py

    Li Liu, Qingle Huang, Sihao Lin, Hongwei Xie, Bing Wang, Xiaojun Chang, Xiaodan Liang: `"Inter-Channel Correlation for Knowledge Distillation" <https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html>`_ @ ICCV 2021 (2021)

    :param feature_pairs: configuration of teacher-student module pairs to compute the L2 distance between the inter-channel correlation matrices of the student and the teacher.
    :type feature_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`ChSimLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Student4ICKD`.

        criterion:
          key: 'ChSimLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'embed_dict.embed1'
                weight: 1
    """

    def __init__(self, feature_pairs, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs
        self.smooth_l1_loss = nn.SmoothL1Loss()

    @staticmethod
    def batch_loss(f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)
        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        g_diff = emd_s - emd_t
        loss = (g_diff * g_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        chsim_loss = 0
        for pair_name, pair_config in self.feature_pairs.items():
            teacher_outputs = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            student_outputs = _extract_feature_map(student_io_dict, pair_config['student'])
            weight = pair_config.get('weight', 1)
            loss = self.batch_loss(student_outputs, teacher_outputs)
            chsim_loss += weight * loss
        return chsim_loss


@register_mid_level_loss
class DISTLoss(nn.Module):
    """
    A loss module for Knowledge Distillation from A Stronger Teacher (DIST).
    Referred to https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py

    Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu: `"Knowledge Distillation from A Stronger Teacher" <https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html>`_ @ NeurIPS 2022 (2022)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param beta: balancing factor for inter-loss.
    :type beta: float
    :param gamma: balancing factor for intra-loss.
    :type gamma: float
    :param tau: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type tau: float
    :param eps: small value to avoid division by zero in cosine simularity.
    :type eps: float
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 beta=1.0, gamma=1.0, tau=1.0, eps=1e-8, **kwargs):
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = eps

    @staticmethod
    def pearson_correlation(y_s, y_t, eps):
        return cosine_similarity(y_s - y_s.mean(1).unsqueeze(1), y_t - y_t.mean(1).unsqueeze(1), eps=eps)

    def inter_class_relation(self, y_s, y_t):
        return 1 - self.pearson_correlation(y_s, y_t, self.eps).mean()

    def intra_class_relation(self, y_s, y_t):
        return self.inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        y_s = (student_logits / self.tau).softmax(dim=1)
        y_t = (teacher_logits / self.tau).softmax(dim=1)
        inter_loss = self.tau ** 2 * self.inter_class_relation(y_s, y_t)
        intra_loss = self.tau ** 2 * self.intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


@register_mid_level_loss
class SRDLoss(nn.Module):
    """
    A loss module for Understanding the Role of the Projector in Knowledge Distillation.
    Referred to https://github.com/roymiles/Simple-Recipe-Distillation/blob/main/imagenet/torchdistill/losses/single.py

    Roy Miles, Krystian Mikolajczyk: `"Understanding the Role of the Projector in Knowledge Distillation" <https://arxiv.org/abs/2303.11098>`_ @ AAAI 2024 (2024)

    :param student_feature_module_path: student model's feature module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SRDModelWrapper`.
    :type student_feature_module_path: str
    :param student_feature_module_io: 'input' or 'output' of the feature module in the student model.
    :type student_feature_module_io: str
    :param teacher_feature_module_path: teacher model's feature module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SRDModelWrapper`.
    :type teacher_feature_module_path: str
    :param teacher_feature_module_io: 'input' or 'output' of the feature module in the teacher model.
    :type teacher_feature_module_io: str
    :param student_linear_module_path: student model's linear module path.
    :type student_linear_module_path: str
    :param student_linear_module_io: 'input' or 'output' of the linear module in the student model.
    :type student_linear_module_io: str
    :param teacher_linear_module_path: teacher model's linear module path.
    :type teacher_linear_module_path: str
    :param teacher_linear_module_io: 'input' or 'output' of the linear module in the teacher model.
    :type teacher_linear_module_io: str
    :param exponent: exponent for feature distillation loss.
    :type exponent: float
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param reduction: loss reduction type.
    :type reduction: str or None
    """

    def __init__(self, student_feature_module_path, student_feature_module_io,
                 teacher_feature_module_path, teacher_feature_module_io,
                 student_linear_module_path, student_linear_module_io,
                 teacher_linear_module_path, teacher_linear_module_io,
                 exponent=1.0, temperature=1.0, reduction='batchmean', **kwargs):
        super().__init__()
        self.student_feature_module_path = student_feature_module_path
        self.student_feature_module_io = student_feature_module_io
        self.teacher_feature_module_path = teacher_feature_module_path
        self.teacher_feature_module_io = teacher_feature_module_io
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.exponent = exponent
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features = student_io_dict[self.student_feature_module_path][self.student_feature_module_io]
        teacher_features = teacher_io_dict[self.teacher_feature_module_path][self.teacher_feature_module_io]
        diff_features = torch.abs(student_features - teacher_features)
        feat_distill_loss = torch.log(diff_features.pow(self.exponent).sum())

        student_logits = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_logits = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        kl_loss = self.criterion(torch.log_softmax(student_logits / self.temperature, dim=1),
                                 torch.softmax(teacher_logits / self.temperature, dim=1))
        loss = 2 * feat_distill_loss + kl_loss
        return loss


@register_mid_level_loss
class LogitStdKDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation (KD) loss module with logits standardization.

    Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao: `"Logit Standardization in Knowledge Distillation" <https://arxiv.org/abs/2403.01427>`_ @ CVPR 2024 (2024)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param eps: value added to the denominator for numerical stability.
    :type eps: float
    :param alpha: balancing factor for :math:`L_{CE}`, cross-entropy.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`) for :math:`L_{KL}`, KL divergence between class-probability distributions softened by :math:`\\tau`.
    :type beta: float or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None
    """
    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 temperature, eps=1e-7, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def standardize(self, logits):
        return (logits - logits.mean(dim=-1, keepdims=True)) / (self.eps + logits.std(dim=-1, keepdims=True))

    def forward(self, student_io_dict, teacher_io_dict, targets=None, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        soft_loss = super().forward(torch.log_softmax(self.standardize(student_logits) / self.temperature, dim=1),
                                    torch.softmax(self.standardize(teacher_logits) / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


@register_mid_level_loss
class DISTPlusLoss(DISTLoss):
    """
    A loss module for DIST+.

    Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu: `"DIST+: Knowledge Distillation From a Stronger Adaptive Teacher" <https://ieeexplore.ieee.org/document/10938241>`_ @ TPAMI (2025)

    :param student_logit_module_path: student model's logit module path.
    :type student_logit_module_path: str
    :param student_logit_module_io: 'input' or 'output' of the module in the student model.
    :type student_logit_module_io: str
    :param teacher_logit_module_path: teacher model's logit module path.
    :type teacher_logit_module_path: str
    :param teacher_logit_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_logit_module_io: str
    :param student_feature_module_path: student model's feature map module path.
    :type student_feature_module_path: str
    :param student_feature_module_io: 'input' or 'output' of the module in the student model.
    :type student_feature_module_io: str
    :param teacher_feature_module_path: teacher model's feature map module path.
    :type teacher_feature_module_path: str
    :param teacher_feature_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_feature_module_io: str
    :param beta: balancing factor for inter-loss.
    :type beta: float
    :param iota: balancing factor for intra-loss.
    :type iota: float
    :param kappa: balancing factor for channel relation loss.
    :type kappa: float
    :param gamma: balancing factor for spatial relation loss.
    :type gamma: float
    :param tau: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type tau: float
    :param eps: small value to avoid division by zero in cosine simularity.
    :type eps: float
    """

    def __init__(self, student_logit_module_path, student_logit_module_io,
                 teacher_logit_module_path, teacher_logit_module_io,
                 student_feature_module_path, student_feature_module_io,
                 teacher_feature_module_path, teacher_feature_module_io,
                 beta=1.0, gamma=1.0, iota=1.0, kappa=1.0, tau=1.0, eps=1e-8, **kwargs):
        super().__init__(
            student_logit_module_path, student_logit_module_io, teacher_logit_module_path, teacher_logit_module_io,
            beta=beta, gamma=gamma, tau=tau, eps=eps
        )
        self.student_feature_module_path = student_feature_module_path
        self.student_feature_module_io = student_feature_module_io
        self.teacher_feature_module_path = teacher_feature_module_path
        self.teacher_feature_module_io = teacher_feature_module_io
        self.iota = iota
        self.kappa = kappa

    def channel_relation(self, f_s, f_t):
        c_mean_f_s = torch.mean(f_s, dim=1, keepdim=True)
        c_mean_f_t = torch.mean(f_t, dim=1, keepdim=True)
        c_centered_f_s = f_s - c_mean_f_s
        c_centered_f_t = f_t - c_mean_f_t
        numerator = torch.sum(c_centered_f_s * c_centered_f_t, dim=1)
        denominator = torch.sqrt(torch.sum(c_centered_f_s ** 2, dim=1) * torch.sum(c_centered_f_t ** 2, dim=1))
        return (numerator / (denominator + self.eps)).mean()

    def spatial_relation(self, f_s, f_t):
        aggregated_f_s = torch.sum(f_s, dim=1).flatten(1)
        aggregated_f_t = torch.sum(f_t, dim=1).flatten(1)
        return self.inter_class_relation(aggregated_f_s.transpose(0, 1), aggregated_f_t.transpose(0, 1))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        dist_loss = super().forward(student_io_dict, teacher_io_dict, *args, **kwargs)
        student_features = student_io_dict[self.student_feature_module_path][self.student_feature_module_io]
        teacher_features = teacher_io_dict[self.teacher_feature_module_path][self.teacher_feature_module_io]
        channel_relation_loss = self.channel_relation(student_features, teacher_features)
        spatial_relation_loss = self.spatial_relation(student_features, teacher_features)
        loss = dist_loss + self.iota * channel_relation_loss + self.kappa * spatial_relation_loss
        return loss

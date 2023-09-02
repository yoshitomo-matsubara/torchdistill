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
       :caption: An example yaml of ``kwargs`` to instantiate :class:`SimpleLossWrapper`.

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
        return self.mid_level_loss.__str__()


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
       :caption: An example yaml of ``kwargs`` and ``weights`` to instantiate :class:`DictLossWrapper` for deeplabv3_resnet50 in torchvision, whose default output is a dict of outputs from its main and auxiliary branches with keys 'out' and 'aux' respectively.

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
        return str(self.weights) + ' * ' + self.mid_level_loss.__str__()


@register_mid_level_loss
class KDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation loss module.

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
    :param alpha: balancing factor.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`).
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
    "A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"
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
    "Paying More Attention to Attention: Improving the Performance of
     Convolutional Neural Networks via Attention Transfer"
    Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
    Discrepancy between Eq. (2) in the paper and the author's implementation
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23
    as partly pointed out at https://github.com/szagoruyko/attention-transfer/issues/34
    To follow the equations in the paper, use mode='paper' in place of 'code'
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
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    Refactored https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py
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
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
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
    "Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"
    Refactored https://github.com/bhheo/AB_distillation/blob/master/cifar10_AB_distillation.py
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
    "Relational Knowledge Distillation"
    Refactored https://github.com/lenscloth/RKD/blob/master/metric/loss.py
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
    "Variational Information Distillation for Knowledge Transfer"
    Referred to https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py
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
    "Correlation Congruence for Knowledge Distillation"
    Configure KDLoss in a yaml file to meet eq. (7), using WeightedSumLoss
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
    "Similarity-Preserving Knowledge Distillation"
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
    "Contrastive Representation Distillation"
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py
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
        """ Draw n samples from multinomial """
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
        """
        pos_idx: the indices of these positive samples in the dataset, size [batch_size]
        contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
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
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
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
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
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
    "Prime-Aware Adaptive Distillation"
    """
    def __init__(self, student_embed_module_path, teacher_embed_module_path,
                 student_embed_module_io='output', teacher_embed_module_io='output',
                 module_path='var_estimator', module_io='output', eps=1e-6, reduction='sum', **kwargs):
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
        return squared_losses.mean()


@register_mid_level_loss
class HierarchicalContextLoss(nn.Module):
    """
    "Distilling Knowledge via Knowledge Review"
    Referred to https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py
    """
    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 reduction='mean', kernel_sizes=None, **kwargs):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [4, 2, 1]

        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.criteria = nn.MSELoss(reduction=reduction)
        self.kernel_sizes = kernel_sizes

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features, _ = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_features = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        _, _, h, _ = student_features.shape
        loss = self.criteria(student_features, teacher_features)
        weight = 1.0
        total_weight = 1.0
        for k in self.kernel_sizes:
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
    "Knowledge Adaptation for Efficient Semantic Segmentation"
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
    "Knowledge Adaptation for Efficient Semantic Segmentation"
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

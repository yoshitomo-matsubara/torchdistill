import torch
from torch import nn
from torch.nn.functional import adaptive_max_pool2d, normalize

from myutils.pytorch import func_util

SINGLE_LOSS_CLASS_DICT = dict()
CUSTOM_LOSS_CLASS_DICT = dict()
FUNC2EXTRACT_ORG_OUTPUT_DICT = dict()


def register_single_loss(cls):
    SINGLE_LOSS_CLASS_DICT[cls.__name__] = cls
    return cls


def register_custom_loss(cls):
    CUSTOM_LOSS_CLASS_DICT[cls.__name__] = cls
    return cls


class SimpleLossWrapper(nn.Module):
    def __init__(self, single_loss, params_config):
        super().__init__()
        self.single_loss = single_loss
        input_config = params_config['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io']
        target_config = params_config['target']
        self.is_target_from_teacher = target_config['is_from_teacher']
        self.target_module_path = target_config['module_path']
        self.target_key = target_config['io']

    @staticmethod
    def extract_value(io_dict, path, key):
        return io_dict[path][key]

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict,
                                         self.input_module_path, self.input_key)
        target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict,
                                          self.target_module_path, self.target_key)
        return self.single_loss(input_batch, target_batch, *args, **kwargs)


@register_single_loss
class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_output, teacher_output, labels=None):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or labels is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(student_output, labels)
        return self.alpha * hard_loss + (1 - self.alpha) * (self.temperature ** 2) * soft_loss


@register_single_loss
class FSPLoss(nn.Module):
    """
    "A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"
    """
    def __init__(self, fsp_pairs, **kwargs):
        super().__init__()
        self.fsp_pairs = fsp_pairs

    @staticmethod
    def extract_feature_map(io_dict, feature_map_config):
        key = list(feature_map_config.keys())[0]
        return io_dict[feature_map_config[key]][key]

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

    def forward(self, student_io_dict, teacher_io_dict):
        fsp_loss = 0
        batch_size = None
        for pair_name, pair_config in self.fsp_pairs.items():
            student_first_feature_map = self.extract_feature_map(student_io_dict, pair_config['student_first'])
            student_second_feature_map = self.extract_feature_map(student_io_dict, pair_config['student_second'])
            student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
            teacher_first_feature_map = self.extract_feature_map(teacher_io_dict, pair_config['teacher_first'])
            teacher_second_feature_map = self.extract_feature_map(teacher_io_dict, pair_config['teacher_second'])
            teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
            factor = pair_config.get('factor', 1)
            fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
            if batch_size is None:
                batch_size = student_first_feature_map.shape[0]
        return fsp_loss / batch_size


@register_single_loss
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

    def forward(self, student_io_dict, teacher_io_dict):
        student_penultimate_outputs = teacher_io_dict[self.student_module_path][self.student_module_io]
        teacher_penultimate_outputs = student_io_dict[self.teacher_module_path][self.teacher_module_io]
        return self.cosine_similarity_loss(student_penultimate_outputs, teacher_penultimate_outputs)


@register_single_loss
class FTLoss(nn.Module):
    """
    "Paraphrasing Complex Network: Network Compression via Factor Transfer"
    """
    def __init__(self, p=1, reduction='batchmean', paraphraser_path='paraphraser',
                 translator_path='translator', **kwargs):
        super().__init__()
        self.norm_loss = nn.L1Loss() if p == 1 else nn.MSELoss()
        self.paraphraser_path = paraphraser_path
        self.translator_path = translator_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict):
        paraphraser_flat_outputs = teacher_io_dict[self.paraphraser_path]['output'].flatten(1)
        translator_flat_outputs = student_io_dict[self.translator_path]['output'].flatten(1)
        batch_size = paraphraser_flat_outputs.shape[0]
        ft_loss = self.norm_loss(paraphraser_flat_outputs / paraphraser_flat_outputs.norm(dim=1).unsqueeze(1),
                                 translator_flat_outputs / translator_flat_outputs.norm(dim=1).unsqueeze(1))
        return ft_loss / batch_size if self.reduction == 'batchmean' else ft_loss


@register_single_loss
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

    def forward(self, student_io_dict, teacher_io_dict):
        teacher_flat_outputs = teacher_io_dict[self.teacher_output_path]['output'].flatten(1)
        student_flat_outputs = student_io_dict[self.student_output_path]['output'].flatten(1)
        rkd_distance_loss = self.compute_rkd_distance_loss(teacher_flat_outputs, student_flat_outputs)
        rkd_angle_loss = self.compute_rkd_angle_loss(teacher_flat_outputs, student_flat_outputs)
        return self.dist_factor * rkd_distance_loss + self.angle_factor * rkd_angle_loss


@register_single_loss
class CCKDLoss(nn.Module):
    """
    "Correlation Congruence for Knowledge Distillation"
    Configure KDLoss in a yaml file to meet eq. (7), using GeneralizedCustomLoss
    """
    def __init__(self, student_linear_path, teacher_linear_path, reduction, **kwargs):
        super().__init__()
        self.student_linear_path = student_linear_path
        self.teacher_linear_path = teacher_linear_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict):
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_path]['output']
        student_linear_outputs = student_io_dict[self.student_linear_path]['output']
        batch_size = teacher_linear_outputs.shape[0]
        teacher_cc = torch.matmul(teacher_linear_outputs, torch.t(teacher_linear_outputs))
        student_cc = torch.matmul(student_linear_outputs, torch.t(student_linear_outputs))
        cc_loss = torch.dist(student_cc, teacher_cc, 2)
        return cc_loss / (batch_size ** 2) if self.reduction == 'batchmean' else cc_loss


@register_single_loss
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
        mat = normalize(torch.matmul(z, torch.t(z)), 2)
        return normalize(mat)

    def compute_spkd_loss(self, teacher_output, student_output):
        g_t = self.matmul_and_normalize(teacher_output)
        g_s = self.matmul_and_normalize(student_output)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_io_dict, teacher_io_dict):
        teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = teacher_outputs.shape[0]
        spkd_losses = [self.compute_spkd_loss(teacher_output, student_output)
                       for teacher_output, student_output in zip(teacher_outputs, student_outputs)]
        spkd_loss = sum(spkd_losses)
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss


def get_single_loss(single_criterion_config, params_config=None):
    loss_type = single_criterion_config['type']
    single_loss = SINGLE_LOSS_CLASS_DICT[loss_type](**single_criterion_config['params']) \
        if loss_type in SINGLE_LOSS_CLASS_DICT else func_util.get_loss(loss_type, single_criterion_config['params'])
    return single_loss if params_config is None else SimpleLossWrapper(single_loss, params_config)


class CustomLoss(nn.Module):
    def __init__(self, criterion_config):
        super().__init__()
        term_dict = dict()
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            for loss_name, loss_config in sub_terms_config.items():
                sub_criterion_config = loss_config['criterion']
                sub_criterion = get_single_loss(sub_criterion_config, loss_config.get('params', None))
                term_dict[loss_name] = (sub_criterion, loss_config['factor'])
        self.term_dict = term_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')


@register_custom_loss
class GeneralizedCustomLoss(CustomLoss):
    def __init__(self, criterion_config):
        super().__init__(criterion_config)
        self.org_loss_factor = criterion_config['org_term'].get('factor', None)

    def forward(self, output_dict, org_loss_dict):
        loss_dict = dict()
        student_output_dict = output_dict['student']
        teacher_output_dict = output_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_output_dict, teacher_output_dict)

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.org_loss_factor is None or self.org_loss_factor == 0:
            return sub_total_loss
        return sub_total_loss + self.org_loss_factor * sum(org_loss_dict.values() if len(org_loss_dict) > 0 else [])


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))


def register_func2extract_org_output(func):
    FUNC2EXTRACT_ORG_OUTPUT_DICT[func.__name__] = func
    return func


@register_func2extract_org_output
def extract_simple_org_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        # Models with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            if uses_teacher_output:
                for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                    org_loss_dict[i] = org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = org_criterion(sub_outputs, targets)
        else:
            org_loss = org_criterion(student_outputs, teacher_outputs, targets) if uses_teacher_output \
                else org_criterion(student_outputs, targets)
            org_loss_dict = {0: org_loss}
    return org_loss_dict


def get_func2extract_org_output(func_name):
    if func_name not in FUNC2EXTRACT_ORG_OUTPUT_DICT:
        return extract_simple_org_loss
    return FUNC2EXTRACT_ORG_OUTPUT_DICT[func_name]

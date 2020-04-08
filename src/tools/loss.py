import torch
from torch import nn

from myutils.pytorch import func_util

SINGLE_LOSS_CLASS_DICT = dict()
CUSTOM_LOSS_CLASS_DICT = dict()


def register_single_loss(cls):
    SINGLE_LOSS_CLASS_DICT[cls.__name__] = cls


def register_custom_loss(cls):
    CUSTOM_LOSS_CLASS_DICT[cls.__name__] = cls


@register_single_loss
class KDLoss(nn.KLDivLoss):
    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
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


def get_single_loss(single_criterion_config):
    loss_type = single_criterion_config['type']
    if loss_type in SINGLE_LOSS_CLASS_DICT:
        return SINGLE_LOSS_CLASS_DICT[loss_type](**single_criterion_config['params'])
    return func_util.get_loss(loss_type, single_criterion_config['params'])


class CustomLoss(nn.Module):
    def __init__(self, criterion_config):
        super().__init__()
        term_dict = dict()
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            for loss_name, loss_config in sub_terms_config.items():
                sub_criterion_config = loss_config['criterion']
                sub_criterion = func_util.get_loss(sub_criterion_config['type'], sub_criterion_config['params'])
                term_dict[loss_name] = (loss_config['ts_modules'], sub_criterion, loss_config['factor'])
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
        for loss_name, ((teacher_path, teacher_output), (student_path, student_output)) in output_dict.items():
            _, criterion, factor = self.term_dict[loss_name]
            loss_dict[loss_name] = criterion(teacher_output, student_output) * factor

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.org_loss_factor is None or self.org_loss_factor == 0:
            return sub_total_loss
        return sub_total_loss + self.org_loss_factor * sum(org_loss_dict.values() if len(org_loss_dict) > 0 else [])


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))

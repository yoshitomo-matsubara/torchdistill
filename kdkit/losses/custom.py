from torch import nn

from kdkit.common.constant import def_logger
from kdkit.losses.single import get_single_loss

CUSTOM_LOSS_CLASS_DICT = dict()

logger = def_logger.getChild(__name__)


def register_custom_loss(cls):
    CUSTOM_LOSS_CLASS_DICT[cls.__name__] = cls
    return cls


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

    def forward(self, output_dict, org_loss_dict, targets):
        loss_dict = dict()
        student_output_dict = output_dict['student']
        teacher_output_dict = output_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_output_dict, teacher_output_dict, targets)

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.org_loss_factor is None or self.org_loss_factor == 0:
            return sub_total_loss
        return sub_total_loss + self.org_loss_factor * sum(org_loss_dict.values() if len(org_loss_dict) > 0 else [])


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))

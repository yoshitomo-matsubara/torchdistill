from torch import nn

from torchdistill.common.constant import def_logger
from torchdistill.losses.single import get_single_loss

CUSTOM_LOSS_CLASS_DICT = dict()

logger = def_logger.getChild(__name__)


def register_custom_loss(arg=None, **kwargs):
    def _register_custom_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        CUSTOM_LOSS_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_custom_loss(arg)
    return _register_custom_loss


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

    def __str__(self):
        desc = 'Loss = '
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for criterion, factor in self.term_dict.values()])
        return desc


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
        if self.org_loss_factor is None or \
                (isinstance(self.org_loss_factor, (int, float)) and self.org_loss_factor == 0):
            return sub_total_loss

        if isinstance(self.org_loss_factor, dict):
            org_loss = sum([self.org_loss_factor[k] * v for k, v in org_loss_dict.items()])
            return sub_total_loss + org_loss
        return sub_total_loss + self.org_loss_factor * sum(org_loss_dict.values() if len(org_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.org_loss_factor, 'OrgLoss')] \
            if self.org_loss_factor is not None and self.org_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))

from torch import nn

from .registry import register_high_level_loss, get_single_loss
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


class AbstractLoss(nn.Module):
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
        raise NotImplementedError('forward function is not implemented')


@register_high_level_loss
class WeightedSumLoss(AbstractLoss):
    def __init__(self, criterion_config):
        super().__init__(criterion_config)
        self.model_loss_factor = criterion_config.get('model_term', dict()).get('factor', None)

    def forward(self, io_dict, model_loss_dict, targets):
        loss_dict = dict()
        student_io_dict = io_dict['student']
        teacher_io_dict = io_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_io_dict, teacher_io_dict, targets)

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.model_loss_factor is None or \
                (isinstance(self.model_loss_factor, (int, float)) and self.model_loss_factor == 0):
            return sub_total_loss

        if isinstance(self.model_loss_factor, dict):
            model_loss = sum([self.model_loss_factor[k] * v for k, v in model_loss_dict.items()])
            return sub_total_loss + model_loss
        return sub_total_loss + self.model_loss_factor * sum(model_loss_dict.values() if len(model_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.model_loss_factor, 'ModelLoss')] \
            if self.model_loss_factor is not None and self.model_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc

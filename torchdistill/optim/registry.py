from torch import nn

from torchdistill.common import misc_util

OPTIM_DICT = misc_util.get_classes_as_dict('torch.optim')
SCHEDULER_DICT = misc_util.get_classes_as_dict('torch.optim.lr_scheduler')


def register_optimizer(cls_or_func):
    OPTIM_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func


def register_scheduler(cls_or_func):
    SCHEDULER_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func


def get_optimizer(target, optim_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()

    is_module = isinstance(target, nn.Module)
    params = target.parameters() if is_module else target
    lower_optim_type = optim_type.lower()
    if lower_optim_type in OPTIM_DICT:
        optim_cls = OPTIM_DICT[lower_optim_type]
        if is_module:
            return optim_cls([p for p in params if p.requires_grad], **param_dict, **kwargs)
        return optim_cls(params, **param_dict, **kwargs)
    raise ValueError('optim_type `{}` is not expected'.format(optim_type))


def get_scheduler(optimizer, scheduler_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()

    lower_scheduler_type = scheduler_type.lower()
    if lower_scheduler_type in SCHEDULER_DICT:
        return SCHEDULER_DICT[lower_scheduler_type](optimizer, **param_dict, **kwargs)
    raise ValueError('scheduler_type `{}` is not expected'.format(scheduler_type))

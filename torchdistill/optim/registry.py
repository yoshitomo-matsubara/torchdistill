from torch import nn

from ..common import misc_util

OPTIM_DICT = misc_util.get_classes_as_dict('torch.optim')
SCHEDULER_DICT = misc_util.get_classes_as_dict('torch.optim.lr_scheduler')


def register_optimizer(arg=None, **kwargs):
    def _register_optimizer(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        OPTIM_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_optimizer(arg)
    return _register_optimizer


def register_scheduler(arg=None, **kwargs):
    def _register_scheduler(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        SCHEDULER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_scheduler(arg)
    return _register_scheduler


def get_optimizer(module, key, filters_params=True, *args, **kwargs):
    is_module = isinstance(module, nn.Module)
    if key in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[key]
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            return optim_cls_or_func(updatable_params, *args, **kwargs)
        return optim_cls_or_func(module, *args, **kwargs)
    raise ValueError('No optimizer `{}` registered'.format(key))


def get_scheduler(optimizer, key, *args, **kwargs):
    if key in SCHEDULER_DICT:
        return SCHEDULER_DICT[key](optimizer, *args, **kwargs)
    raise ValueError('No scheduler `{}` registered'.format(key))

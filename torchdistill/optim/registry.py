from torch import nn

from torchdistill.common import misc_util

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


def get_optimizer(module, optim_type, param_dict=None, filters_params=True, **kwargs):
    if param_dict is None:
        param_dict = dict()

    is_module = isinstance(module, nn.Module)
    lower_optim_type = optim_type.lower()
    if lower_optim_type in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[lower_optim_type]
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            return optim_cls_or_func(updatable_params, **param_dict, **kwargs)
        return optim_cls_or_func(module, **param_dict, **kwargs)
    raise ValueError('optim_type `{}` is not expected'.format(optim_type))


def get_scheduler(optimizer, scheduler_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()

    lower_scheduler_type = scheduler_type.lower()
    if lower_scheduler_type in SCHEDULER_DICT:
        return SCHEDULER_DICT[lower_scheduler_type](optimizer, **param_dict, **kwargs)
    raise ValueError('scheduler_type `{}` is not expected'.format(scheduler_type))

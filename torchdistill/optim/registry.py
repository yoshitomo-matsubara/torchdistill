from torch import nn

from ..common import misc_util

OPTIM_DICT = misc_util.get_classes_as_dict('torch.optim')
SCHEDULER_DICT = misc_util.get_classes_as_dict('torch.optim.lr_scheduler')


def register_optimizer(arg=None, **kwargs):
    """
    Registers an optimizer class or function to instantiate it.

    :param arg: class or function to be registered as an optimizer.
    :type arg: class or typing.Callable or None
    :return: registered optimizer class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The optimizer will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.optim import Optimizer
        >>> from torchdistill.optim.registry import register_optimizer
        >>>
        >>> @register_optimizer(key='my_custom_optimizer')
        >>> class CustomOptimizer(Optimizer):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom optimizer class')

        In the example, ``CustomOptimizer`` class is registered with a key "my_custom_optimizer".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomOptimizer`` class by
        "my_custom_optimizer".
    """
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
    """
    Registers a scheduler class or function to instantiate it.

    :param arg: class or function to be registered as a scheduler.
    :type arg: class or typing.Callable or None
    :return: registered scheduler class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The scheduler will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.optim.lr_scheduler import LRScheduler
        >>> from torchdistill.optim.registry import register_scheduler
        >>>
        >>> @register_scheduler(key='my_custom_scheduler')
        >>> class CustomScheduler(LRScheduler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom scheduler class')

        In the example, ``CustomScheduler`` class is registered with a key "my_custom_scheduler".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomScheduler`` class by
        "my_custom_scheduler".
    """
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
    """
    Gets an optimizer from the optimizer registry.

    :param module: module to be added to optimizer.
    :type module: nn.Module
    :param key: optimizer key.
    :type key: str
    :param filters_params: if True, filers out parameters whose `required_grad = False`.
    :type filters_params: bool
    :return: optimizer.
    :rtype: Optimizer
    """
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
    """
    Gets a scheduler from the scheduler registry.

    :param optimizer: optimizer to be added to scheduler.
    :type optimizer: Optimizer
    :param key: scheduler key.
    :type key: str
    :return: scheduler.
    :rtype: LRScheduler
    """
    if key in SCHEDULER_DICT:
        return SCHEDULER_DICT[key](optimizer, *args, **kwargs)
    raise ValueError('No scheduler `{}` registered'.format(key))

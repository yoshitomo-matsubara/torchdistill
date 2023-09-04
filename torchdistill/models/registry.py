import torch

from ..common import misc_util

MODEL_DICT = dict()
MODEL_DICT = dict()
ADAPTATION_MODULE_DICT = dict()
AUXILIARY_MODEL_WRAPPER_DICT = dict()
MODULE_DICT = misc_util.get_classes_as_dict('torch.nn')


def register_model(arg=None, **kwargs):
    """
    Registers a model class or function to instantiate it.

    :param arg: class or function to be registered as a model.
    :type arg: class or typing.Callable or None
    :return: registered model class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The model will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_model
        >>>
        >>> @register_model(key='my_custom_model')
        >>> class CustomModel(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom model class')

        In the example, ``CustomModel`` class is registered with a key "my_custom_model".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomModel`` class by
        "my_custom_model".
    """
    def _register_model(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model(arg)
    return _register_model


def register_adaptation_module(arg=None, **kwargs):
    """
    Registers an adaptation module class or function to instantiate it.

    :param arg: class or function to be registered as an adaptation module.
    :type arg: class or typing.Callable or None
    :return: registered adaptation module class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The adaptation module will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_adaptation_module
        >>>
        >>> @register_adaptation_module(key='my_custom_adaptation_module')
        >>> class CustomAdaptationModule(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom adaptation module class')

        In the example, ``CustomAdaptationModule`` class is registered with a key "my_custom_adaptation_module".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomAdaptationModule`` class by
        "my_custom_adaptation_module".
    """
    def _register_adaptation_module(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        ADAPTATION_MODULE_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_adaptation_module(arg)
    return _register_adaptation_module


def register_auxiliary_model_wrapper(arg=None, **kwargs):
    """
    Registers an auxiliary model wrapper class or function to instantiate it.

    :param arg: class or function to be registered as an auxiliary model wrapper.
    :type arg: class or typing.Callable or None
    :return: registered auxiliary model wrapper class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The auxiliary model wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_auxiliary_model_wrapper
        >>>
        >>> @register_auxiliary_model_wrapper(key='my_custom_auxiliary_model_wrapper')
        >>> class CustomAuxiliaryModelWrapper(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom auxiliary model wrapper class')

        In the example, ``CustomAuxiliaryModelWrapper`` class is registered with a key "my_custom_auxiliary_model_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomAuxiliaryModelWrapper`` class by
        "my_custom_auxiliary_model_wrapper".
    """
    def _register_auxiliary_model_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        AUXILIARY_MODEL_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_auxiliary_model_wrapper(arg)
    return _register_auxiliary_model_wrapper


def get_model(key, repo_or_dir=None, *args, **kwargs):
    """
    Gets a model from the model registry.

    :param key: model key.
    :type key: str
    :param repo_or_dir: ``repo_or_dir`` for torch.hub.load.
    :type repo_or_dir: str or None
    :return: model.
    :rtype: nn.Module
    """
    if key in MODEL_DICT:
        return MODEL_DICT[key](*args, **kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, key, *args, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(key))


def get_adaptation_module(key, *args, **kwargs):
    """
    Gets an adaptation module from the adaptation module registry.

    :param key: model key.
    :type key: str
    :return: adaptation module.
    :rtype: nn.Module
    """
    if key in ADAPTATION_MODULE_DICT:
        return ADAPTATION_MODULE_DICT[key](*args, **kwargs)
    elif key in MODULE_DICT:
        return MODULE_DICT[key](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(key))


def get_auxiliary_model_wrapper(key, *args, **kwargs):
    """
    Gets an auxiliary model wrapper from the auxiliary model wrapper registry.

    :param key: model key.
    :type key: str
    :return: auxiliary model wrapper.
    :rtype: nn.Module
    """
    if key in AUXILIARY_MODEL_WRAPPER_DICT:
        return AUXILIARY_MODEL_WRAPPER_DICT[key](*args, **kwargs)
    raise ValueError('No auxiliary model wrapper `{}` registered'.format(key))

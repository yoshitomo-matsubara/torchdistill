import torch

from ..common import misc_util

MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()
ADAPTATION_MODULE_DICT = dict()
AUXILIARY_MODEL_WRAPPER_DICT = dict()
MODULE_DICT = misc_util.get_classes_as_dict('torch.nn')


def register_model_class(arg=None, **kwargs):
    def _register_model_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model_class(arg)
    return _register_model_class


def register_model_func(arg=None, **kwargs):
    def _register_model_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        MODEL_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_model_func(arg)
    return _register_model_func


def register_adaptation_module(arg=None, **kwargs):
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
    def _register_auxiliary_model_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        AUXILIARY_MODEL_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_auxiliary_model_wrapper(arg)
    return _register_auxiliary_model_wrapper


def get_model(key, repo_or_dir=None, **kwargs):
    if key in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[key](**kwargs)
    elif key in MODEL_FUNC_DICT:
        return MODEL_FUNC_DICT[key](**kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, key, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(key))


def get_adaptation_module(key, *args, **kwargs):
    if key in ADAPTATION_MODULE_DICT:
        return ADAPTATION_MODULE_DICT[key](*args, **kwargs)
    elif key in MODULE_DICT:
        return MODULE_DICT[key](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(key))


def get_auxiliary_model_wrapper(key, *args, **kwargs):
    if key in AUXILIARY_MODEL_WRAPPER_DICT:
        return AUXILIARY_MODEL_WRAPPER_DICT[key](*args, **kwargs)
    raise ValueError('No special module `{}` registered'.format(key))

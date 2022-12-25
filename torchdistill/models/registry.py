import torch
from torch import nn


MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()
ADAPTATION_CLASS_DICT = dict()
SPECIAL_CLASS_DICT = dict()
MODULE_CLASS_DICT = nn.__dict__


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
    def _register_adaptation_module(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        ADAPTATION_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_adaptation_module(arg)
    return _register_adaptation_module


def register_special_module(arg=None, **kwargs):
    def _register_special_module(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SPECIAL_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_special_module(arg)
    return _register_special_module


def get_model(model_name, repo_or_dir=None, **kwargs):
    if model_name in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[model_name](**kwargs)
    elif model_name in MODEL_FUNC_DICT:
        return MODEL_FUNC_DICT[model_name](**kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, model_name, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(model_name))


def get_adaptation_module(class_name, *args, **kwargs):
    if class_name in ADAPTATION_CLASS_DICT:
        return ADAPTATION_CLASS_DICT[class_name](*args, **kwargs)
    elif class_name in MODULE_CLASS_DICT:
        return MODULE_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(class_name))


def get_special_module(class_name, *args, **kwargs):
    if class_name in SPECIAL_CLASS_DICT:
        return SPECIAL_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No special module `{}` registered'.format(class_name))

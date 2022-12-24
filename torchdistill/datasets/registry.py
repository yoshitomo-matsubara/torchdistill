from types import BuiltinFunctionType, BuiltinMethodType, FunctionType

import torch
import torchvision

DATASET_DICT = dict()
COLLATE_FUNC_DICT = dict()
SAMPLE_LOADER_CLASS_DICT = dict()
SAMPLE_LOADER_FUNC_DICT = dict()
BATCH_SAMPLER_CLASS_DICT = dict()
TRANSFORM_CLASS_DICT = dict()
WRAPPER_CLASS_DICT = dict()
DATASET_DICT.update(torchvision.datasets.__dict__)
BATCH_SAMPLER_CLASS_DICT.update(torch.utils.data.sampler.__dict__)


def register_dataset(arg=None, **kwargs):
    def _register_dataset(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset(arg)
    return _register_dataset


def register_collate_func(arg=None, **kwargs):
    def _register_collate_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__ if isinstance(func, (BuiltinMethodType, BuiltinFunctionType, FunctionType)) \
                else type(func).__name__

        COLLATE_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_collate_func(arg)
    return _register_collate_func


def register_sample_loader_class(arg=None, **kwargs):
    def _register_sample_loader_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SAMPLE_LOADER_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_sample_loader_class(arg)
    return _register_sample_loader_class


def register_sample_loader_func(arg=None, **kwargs):
    def _register_sample_loader_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        SAMPLE_LOADER_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_sample_loader_func(arg)
    return _register_sample_loader_func


def register_batch_sampler_class(arg=None, **kwargs):
    def _register_batch_sampler_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        BATCH_SAMPLER_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_batch_sampler_class(arg)
    return _register_batch_sampler_class


def register_transform_class(arg=None, **kwargs):
    def _register_transform_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        TRANSFORM_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_transform_class(arg)
    return _register_transform_class


def register_dataset_wrapper(arg=None, **kwargs):
    def _register_dataset_wrapper(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        WRAPPER_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_dataset_wrapper(arg)
    return _register_dataset_wrapper


def get_collate_func(func_name):
    if func_name is None:
        return None
    elif func_name in COLLATE_FUNC_DICT:
        return COLLATE_FUNC_DICT[func_name]
    raise ValueError('No collate function `{}` registered'.format(func_name))


def get_sample_loader(obj_name, *args, **kwargs):
    if obj_name is None:
        return None
    elif obj_name in SAMPLE_LOADER_CLASS_DICT:
        return SAMPLE_LOADER_CLASS_DICT[obj_name](*args, **kwargs)
    elif obj_name in SAMPLE_LOADER_FUNC_DICT:
        return SAMPLE_LOADER_FUNC_DICT[obj_name]
    raise ValueError('No sample loader `{}` registered.'.format(obj_name))


def get_batch_sampler(class_name, *args, **kwargs):
    if class_name is None:
        return None

    if class_name not in BATCH_SAMPLER_CLASS_DICT and class_name != 'BatchSampler':
        raise ValueError('No batch sampler `{}` registered.'.format(class_name))

    batch_sampler_cls = BATCH_SAMPLER_CLASS_DICT[class_name]
    return batch_sampler_cls(*args, **kwargs)


def get_transform(obj_name, *args, **kwargs):
    if obj_name in TRANSFORM_CLASS_DICT:
        return TRANSFORM_CLASS_DICT[obj_name](*args, **kwargs)
    raise ValueError('No transform `{}` registered.'.format(obj_name))


def get_dataset_wrapper(class_name, *args, **kwargs):
    if class_name not in WRAPPER_CLASS_DICT:
        return WRAPPER_CLASS_DICT[class_name](*args, **kwargs)
    raise ValueError('No dataset wrapper `{}` registered.'.format(class_name))

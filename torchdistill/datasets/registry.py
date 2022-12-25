from types import BuiltinFunctionType, BuiltinMethodType, FunctionType

from ..common import misc_util

DATASET_DICT = dict()
COLLATE_FUNC_DICT = dict()
SAMPLE_LOADER_CLASS_DICT = dict()
SAMPLE_LOADER_FUNC_DICT = dict()
BATCH_SAMPLER_DICT = dict()
TRANSFORM_DICT = dict()
DATASET_WRAPPER_DICT = dict()

DATASET_DICT.update(misc_util.get_classes_as_dict('torchvision.datasets'))
BATCH_SAMPLER_DICT.update(misc_util.get_classes_as_dict('torch.utils.data.sampler'))


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


def register_batch_sampler(arg=None, **kwargs):
    def _register_batch_sampler(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        BATCH_SAMPLER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_batch_sampler(arg)
    return _register_batch_sampler


def register_transform(arg=None, **kwargs):
    def _register_transform(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        TRANSFORM_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_transform(arg)
    return _register_transform


def register_dataset_wrapper(arg=None, **kwargs):
    def _register_dataset_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset_wrapper(arg)
    return _register_dataset_wrapper


def get_collate_func(key):
    if key is None:
        return None
    elif key in COLLATE_FUNC_DICT:
        return COLLATE_FUNC_DICT[key]
    raise ValueError('No collate function `{}` registered'.format(key))


def get_sample_loader(key, *args, **kwargs):
    if key is None:
        return None
    elif key in SAMPLE_LOADER_CLASS_DICT:
        return SAMPLE_LOADER_CLASS_DICT[key](*args, **kwargs)
    elif key in SAMPLE_LOADER_FUNC_DICT:
        return SAMPLE_LOADER_FUNC_DICT[key]
    raise ValueError('No sample loader `{}` registered.'.format(key))


def get_batch_sampler(key, *args, **kwargs):
    if key is None:
        return None

    if key not in BATCH_SAMPLER_DICT and key != 'BatchSampler':
        raise ValueError('No batch sampler `{}` registered.'.format(key))

    batch_sampler_cls = BATCH_SAMPLER_DICT[key]
    return batch_sampler_cls(*args, **kwargs)


def get_transform(key, *args, **kwargs):
    if key in TRANSFORM_DICT:
        return TRANSFORM_DICT[key](*args, **kwargs)
    raise ValueError('No transform `{}` registered.'.format(key))


def get_dataset_wrapper(key, *args, **kwargs):
    if key not in DATASET_WRAPPER_DICT:
        return DATASET_WRAPPER_DICT[key](*args, **kwargs)
    raise ValueError('No dataset wrapper `{}` registered.'.format(key))

from types import BuiltinFunctionType, BuiltinMethodType, FunctionType

from ..common import misc_util

DATASET_DICT = dict()
COLLATE_FUNC_DICT = dict()
SAMPLE_LOADER_DICT = dict()
BATCH_SAMPLER_DICT = dict()
TRANSFORM_DICT = dict()
DATASET_WRAPPER_DICT = dict()

DATASET_DICT.update(misc_util.get_classes_as_dict('torchvision.datasets'))
BATCH_SAMPLER_DICT.update(misc_util.get_classes_as_dict('torch.utils.data.sampler'))


def register_dataset(arg=None, **kwargs):
    """
    Registers a dataset class or function to instantiate it.

    :param arg: class or function to be registered as a dataset.
    :type arg: class or typing.Callable or None
    :return: registered dataset class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The dataset will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Dataset
        >>> from torchdistill.datasets.registry import register_dataset
        >>> @register_dataset(key='my_custom_dataset')
        >>> class CustomDataset(Dataset):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomDataset`` class is registered with a key "my_custom_dataset".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomDataset`` class by
        "my_custom_dataset".
    """
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
    """
    Registers a collate function.

    :param arg: function to be registered as a collate function.
    :type arg: typing.Callable or None
    :return: registered function.
    :rtype: typing.Callable

    .. note::
        The collate function will be registered as an option.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.datasets.registry import register_collate_func
        >>>
        >>> @register_collate_func(key='my_custom_collate')
        >>> def custom_collate(batch, label):
        >>>     print('This is my custom collate function')
        >>>     return batch, label

        In the example, ``custom_collate`` function is registered with a key "my_custom_collate".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``custom_collate`` function by
        "my_custom_collate".
    """
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


def register_sample_loader(arg=None, **kwargs):
    """
    Registers a sample loader class or function to instantiate it.

    :param arg: class or function to be registered as a sample loader.
    :type arg: class or typing.Callable or None
    :return: registered sample loader class or function to instantiate it.
    :rtype: class

    .. note::
        The sample loader will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Sampler
        >>> from torchdistill.datasets.registry import register_sample_loader
        >>> @register_sample_loader(key='my_custom_sample_loader')
        >>> class CustomSampleLoader(Sampler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomSampleLoader`` class is registered with a key "my_custom_sample_loader".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomSampleLoader`` class by
        "my_custom_sample_loader".
    """
    def _register_sample_loader_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SAMPLE_LOADER_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_sample_loader_class(arg)
    return _register_sample_loader_class


def register_batch_sampler(arg=None, **kwargs):
    """
    Registers a batch sampler or function to instantiate it.

    :param arg: function to be registered as a batch sample loader.
    :type arg: typing.Callable or None
    :return: registered batch sample loader function.
    :rtype: typing.Callable

    .. note::
        The batch sampler will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Sampler
        >>> from torchdistill.datasets.registry import register_batch_sampler
        >>> @register_batch_sampler(key='my_custom_batch_sampler')
        >>> class CustomSampleLoader(Sampler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomSampleLoader`` class is registered with a key "my_custom_batch_sampler".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomSampleLoader`` class by
        "my_custom_batch_sampler".
    """
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
    """
    Registers a transform class or function to instantiate it.

    :param arg: class/function to be registered as a transform.
    :type arg: class or typing.Callable or None
    :return: registered transform class/function.
    :rtype: typing.Callable

    .. note::
        The transform will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.datasets.registry import register_transform
        >>> @register_transform(key='my_custom_transform')
        >>> class CustomTransform(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom transform class')

        In the example, ``CustomTransform`` class is registered with a key "my_custom_transform".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomTransform`` class by
        "my_custom_transform".
    """
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
    """
    Registers a dataset wrapper class or function to instantiate it.

    :param arg: class/function to be registered as a dataset wrapper.
    :type arg: class or typing.Callable or None
    :return: registered dataset wrapper class/function.
    :rtype: typing.Callable

    .. note::
        The dataset wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Dataset
        >>> from torchdistill.datasets.registry import register_dataset_wrapper
        >>> @register_transform(key='my_custom_dataset_wrapper')
        >>> class CustomDatasetWrapper(Dataset):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset wrapper class')

        In the example, ``CustomDatasetWrapper`` class is registered with a key "my_custom_dataset_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomDatasetWrapper`` class by
        "my_custom_dataset_wrapper".
    """
    def _register_dataset_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset_wrapper(arg)
    return _register_dataset_wrapper


def get_dataset(key):
    """
    Gets a registered dataset class or function to instantiate it.

    :param key: unique key to identify the registered dataset class/function.
    :type key: str
    :return: registered dataset class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None
    elif key in DATASET_DICT:
        return DATASET_DICT[key]
    raise ValueError('No dataset `{}` registered'.format(key))


def get_collate_func(key):
    """
    Gets a registered collate function.

    :param key: unique key to identify the registered collate function.
    :type key: str
    :return: registered collate function.
    :rtype: typing.Callable
    """
    if key is None:
        return None
    elif key in COLLATE_FUNC_DICT:
        return COLLATE_FUNC_DICT[key]
    raise ValueError('No collate function `{}` registered'.format(key))


def get_sample_loader(key):
    """
    Gets a registered sample loader class or function to instantiate it.

    :param key: unique key to identify the registered sample loader class or function to instantiate it.
    :type key: str
    :return: registered sample loader class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None
    elif key in SAMPLE_LOADER_DICT:
        return SAMPLE_LOADER_DICT[key]
    raise ValueError('No sample loader `{}` registered.'.format(key))


def get_batch_sampler(key):
    """
    Gets a registered batch sampler class or function to instantiate it.

    :param key: unique key to identify the registered batch sampler class or function to instantiate it.
    :type key: str
    :return: registered batch sampler class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None

    if key not in BATCH_SAMPLER_DICT and key != 'BatchSampler':
        raise ValueError('No batch sampler `{}` registered.'.format(key))
    return BATCH_SAMPLER_DICT[key]


def get_transform(key):
    """
    Gets a registered transform class or function to instantiate it.

    :param key: unique key to identify the registered transform class or function to instantiate it.
    :type key: str
    :return: registered transform class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key in TRANSFORM_DICT:
        return TRANSFORM_DICT[key]
    raise ValueError('No transform `{}` registered.'.format(key))


def get_dataset_wrapper(key):
    """
    Gets a registered dataset wrapper class or function to instantiate it.

    :param key: unique key to identify the registered dataset wrapper class or function to instantiate it.
    :type key: str
    :return: registered dataset wrapper class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key in DATASET_WRAPPER_DICT:
        return DATASET_WRAPPER_DICT[key]
    raise ValueError('No dataset wrapper `{}` registered.'.format(key))

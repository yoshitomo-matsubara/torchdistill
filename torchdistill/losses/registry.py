from ..common import misc_util

LOSS_DICT = misc_util.get_classes_as_dict('torch.nn.modules.loss')
LOW_LEVEL_LOSS_DICT = dict()
MIDDLE_LEVEL_LOSS_DICT = dict()
HIGH_LEVEL_LOSS_DICT = dict()
LOSS_WRAPPER_DICT = dict()
FUNC2EXTRACT_MODEL_OUTPUT_DICT = dict()


def register_low_level_loss(arg=None, **kwargs):
    """
    Registers a low-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a low-level loss.
    :type arg: class or typing.Callable or None
    :return: registered low-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The low-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_low_level_loss
        >>>
        >>> @register_low_level_loss(key='my_custom_low_level_loss')
        >>> class CustomLowLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom low-level loss class')

        In the example, ``CustomLowLevelLoss`` class is registered with a key "my_custom_low_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomLowLevelLoss`` class by
        "my_custom_low_level_loss".
    """
    def _register_low_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        LOW_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_low_level_loss(arg)
    return _register_low_level_loss


def register_mid_level_loss(arg=None, **kwargs):
    """
    Registers a middle-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a middle-level loss.
    :type arg: class or typing.Callable or None
    :return: registered middle-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The middle-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_mid_level_loss
        >>>
        >>> @register_mid_level_loss(key='my_custom_mid_level_loss')
        >>> class CustomMidLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom middle-level loss class')

        In the example, ``CustomMidLevelLoss`` class is registered with a key "my_custom_mid_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomMidLevelLoss`` class by
        "my_custom_mid_level_loss".
    """
    def _register_mid_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        MIDDLE_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_mid_level_loss(arg)
    return _register_mid_level_loss


def register_high_level_loss(arg=None, **kwargs):
    """
    Registers a high-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a high-level loss.
    :type arg: class or typing.Callable or None
    :return: registered high-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The high-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_high_level_loss
        >>>
        >>> @register_high_level_loss(key='my_custom_high_level_loss')
        >>> class CustomHighLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom high-level loss class')

        In the example, ``CustomHighLevelLoss`` class is registered with a key "my_custom_high_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomHighLevelLoss`` class by
        "my_custom_high_level_loss".
    """
    def _register_high_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        HIGH_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_high_level_loss(arg)
    return _register_high_level_loss


def register_loss_wrapper(arg=None, **kwargs):
    """
    Registers a loss wrapper class or function to instantiate it.

    :param arg: class or function to be registered as a loss wrapper.
    :type arg: class or typing.Callable or None
    :return: registered loss wrapper class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The loss wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_loss_wrapper
        >>>
        >>> @register_loss_wrapper(key='my_custom_loss_wrapper')
        >>> class CustomLossWrapper(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom loss wrapper class')

        In the example, ``CustomHighLevelLoss`` class is registered with a key "my_custom_loss_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomLossWrapper`` class by
        "my_custom_loss_wrapper".
    """
    def _register_loss_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        LOSS_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_loss_wrapper(arg)
    return _register_loss_wrapper


def register_func2extract_model_output(arg=None, **kwargs):
    """
    Registers a function to extract model output.

    :param arg: function to be registered for extracting model output.
    :type arg: typing.Callable or None
    :return: registered function.
    :rtype: typing.Callable

    .. note::
        The function to extract model output will be registered as an option.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.losses.registry import register_func2extract_model_output
        >>>
        >>> @register_func2extract_model_output(key='my_custom_function2extract_model_output')
        >>> def custom_func2extract_model_output(batch, label):
        >>>     print('This is my custom collate function')
        >>>     return batch, label

        In the example, ``custom_func2extract_model_output`` function is registered with a key "my_custom_function2extract_model_output".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``custom_func2extract_model_output`` function by
        "my_custom_function2extract_model_output".
    """
    def _register_func2extract_model_output(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FUNC2EXTRACT_MODEL_OUTPUT_DICT[key] = func
        return func

    if callable(arg):
        return _register_func2extract_model_output(arg)
    return _register_func2extract_model_output


def get_low_level_loss(key, **kwargs):
    """
    Gets a registered (low-level) loss module.

    :param key: unique key to identify the registered loss class/function.
    :type key: str
    :return: registered loss class or function to instantiate it.
    :rtype: nn.Module
    """
    if key in LOSS_DICT:
        return LOSS_DICT[key](**kwargs)
    elif key in LOW_LEVEL_LOSS_DICT:
        return LOW_LEVEL_LOSS_DICT[key](**kwargs)
    raise ValueError('No loss `{}` registered'.format(key))


def get_mid_level_loss(mid_level_criterion_config, criterion_wrapper_config=None):
    """
    Gets a registered middle-level loss module.

    :param mid_level_criterion_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type mid_level_criterion_config: dict
    :param criterion_wrapper_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type criterion_wrapper_config: dict
    :return: registered middle-level loss class or function to instantiate it.
    :rtype: nn.Module
    """
    loss_key = mid_level_criterion_config['key']
    mid_level_loss = MIDDLE_LEVEL_LOSS_DICT[loss_key](**mid_level_criterion_config['kwargs']) \
        if loss_key in MIDDLE_LEVEL_LOSS_DICT else get_low_level_loss(loss_key, **mid_level_criterion_config['kwargs'])
    if criterion_wrapper_config is None or len(criterion_wrapper_config) == 0:
        return mid_level_loss
    return get_loss_wrapper(mid_level_loss, criterion_wrapper_config)


def get_high_level_loss(criterion_config):
    """
    Gets a registered high-level loss module.

    :param criterion_config: high-level loss configuration to identify and instantiate the registered high-level loss class.
    :type criterion_config: dict
    :return: registered high-level loss class or function to instantiate it.
    :rtype: nn.Module
    """
    criterion_key = criterion_config['key']
    args = criterion_config.get('args', None)
    kwargs = criterion_config.get('kwargs', None)
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    if criterion_key in HIGH_LEVEL_LOSS_DICT:
        return HIGH_LEVEL_LOSS_DICT[criterion_key](*args, **kwargs)
    raise ValueError('No high-level loss `{}` registered'.format(criterion_key))


def get_loss_wrapper(mid_level_loss, criterion_wrapper_config):
    """
    Gets a registered loss wrapper module.

    :param mid_level_loss: middle-level loss module.
    :type mid_level_loss: nn.Module
    :param criterion_wrapper_config: loss wrapper configuration to identify and instantiate the registered loss wrapper class.
    :type criterion_wrapper_config: dict
    :return: registered loss wrapper class or function to instantiate it.
    :rtype: nn.Module
    """
    wrapper_key = criterion_wrapper_config['key']
    args = criterion_wrapper_config.get('args', None)
    kwargs = criterion_wrapper_config.get('kwargs', None)
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    if wrapper_key in LOSS_WRAPPER_DICT:
        return LOSS_WRAPPER_DICT[wrapper_key](mid_level_loss, *args, **kwargs)
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_key))


def get_func2extract_model_output(key):
    """
    Gets a registered function to extract model output.

    :param key: unique key to identify the registered function to extract model output.
    :type key: str
    :return: registered function to extract model output.
    :rtype: typing.Callable
    """
    if key is None:
        key = 'extract_model_loss_dict'
    if key in FUNC2EXTRACT_MODEL_OUTPUT_DICT:
        return FUNC2EXTRACT_MODEL_OUTPUT_DICT[key]
    raise ValueError('No function to extract original output `{}` registered'.format(key))

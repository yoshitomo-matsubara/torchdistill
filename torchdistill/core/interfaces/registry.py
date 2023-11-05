PRE_EPOCH_PROC_FUNC_DICT = dict()
PRE_FORWARD_PROC_FUNC_DICT = dict()
FORWARD_PROC_FUNC_DICT = dict()
POST_FORWARD_PROC_FUNC_DICT = dict()
POST_EPOCH_PROC_FUNC_DICT = dict()


def register_pre_epoch_proc_func(arg=None, **kwargs):
    """
    Registers a pre-epoch process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a pre-epoch process function.
    :type arg: typing.Callable or None
    :return: registered pre-epoch process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the pre-epoch process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_pre_epoch_proc_func
        >>> @register_pre_epoch_proc_func(key='my_custom_pre_epoch_proc_func')
        >>> def new_pre_epoch_proc(self, epoch=None, **kwargs):
        >>>     print('This is my custom pre-epoch process function')

        In the example, ``new_pre_epoch_proc`` function is registered with a key "my_custom_pre_epoch_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_pre_epoch_proc`` function by
        "my_custom_pre_epoch_proc_func".
    """
    def _register_pre_epoch_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        PRE_EPOCH_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_pre_epoch_proc_func(arg)
    return _register_pre_epoch_proc_func


def register_pre_forward_proc_func(arg=None, **kwargs):
    """
    Registers a pre-forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a pre-forward process function.
    :type arg: typing.Callable or None
    :return: registered pre-forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the pre-forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_pre_forward_proc_func
        >>> @register_pre_forward_proc_func(key='my_custom_pre_forward_proc_func')
        >>> def new_pre_forward_proc(self, *args, **kwargs):
        >>>     print('This is my custom pre-forward process function')

        In the example, ``new_pre_forward_proc`` function is registered with a key "my_custom_pre_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_pre_forward_proc`` function by
        "my_custom_pre_forward_proc_func".
    """
    def _register_pre_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        PRE_FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_pre_forward_proc_func(arg)
    return _register_pre_forward_proc_func


def register_forward_proc_func(arg=None, **kwargs):
    """
    Registers a forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a forward process function.
    :type arg: typing.Callable or None
    :return: registered forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_forward_proc_func
        >>> @register_forward_proc_func(key='my_custom_forward_proc_func')
        >>> def new_forward_proc(model, sample_batch, targets=None, supp_dict=None, **kwargs):
        >>>     print('This is my custom forward process function')

        In the example, ``new_forward_proc`` function is registered with a key "my_custom_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_forward_proc`` function by
        "my_custom_forward_proc_func".
    """
    def _register_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_forward_proc_func(arg)
    return _register_forward_proc_func


def register_post_forward_proc_func(arg=None, **kwargs):
    """
    Registers a post-forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a post-forward process function.
    :type arg: typing.Callable or None
    :return: registered post-forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the post-forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_post_forward_proc_func
        >>> @register_post_forward_proc_func(key='my_custom_post_forward_proc_func')
        >>> def new_post_forward_proc(self, loss, metrics=None, **kwargs):
        >>>     print('This is my custom post-forward process function')

        In the example, ``new_post_forward_proc`` function is registered with a key "my_custom_post_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_post_forward_proc`` function by
        "my_custom_post_forward_proc_func".
    """
    def _register_post_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        POST_FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_post_forward_proc_func(arg)
    return _register_post_forward_proc_func


def register_post_epoch_proc_func(arg=None, **kwargs):
    """
    Registers a post-epoch process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a post-epoch process function.
    :type arg: typing.Callable or None
    :return: registered post-epoch process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the post-epoch process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_post_epoch_proc_func
        >>> @register_post_epoch_proc_func(key='my_custom_post_epoch_proc_func')
        >>> def new_post_epoch_proc(self, metrics=None, **kwargs):
        >>>     print('This is my custom post-epoch process function')

        In the example, ``new_post_epoch_proc`` function is registered with a key "my_custom_post_epoch_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_post_epoch_proc`` function by
        "my_custom_post_epoch_proc_func".
    """
    def _register_post_epoch_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        POST_EPOCH_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_post_epoch_proc_func(arg)
    return _register_post_epoch_proc_func


def get_pre_epoch_proc_func(key):
    """
    Gets a registered pre-epoch process function.

    :param key: unique key to identify the registered pre-epoch process function.
    :type key: str
    :return: registered pre-epoch process function.
    :rtype: typing.Callable
    """
    if key in PRE_EPOCH_PROC_FUNC_DICT:
        return PRE_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No pre-epoch process function `{}` registered'.format(key))


def get_pre_forward_proc_func(key):
    """
    Gets a registered pre-forward process function.

    :param key: unique key to identify the registered pre-forward process function.
    :type key: str
    :return: registered pre-forward process function.
    :rtype: typing.Callable
    """
    if key in PRE_FORWARD_PROC_FUNC_DICT:
        return PRE_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No pre-forward process function `{}` registered'.format(key))


def get_forward_proc_func(key):
    """
    Gets a registered forward process function.

    :param key: unique key to identify the registered forward process function.
    :type key: str
    :return: registered forward process function.
    :rtype: typing.Callable
    """
    if key in FORWARD_PROC_FUNC_DICT:
        return FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No forward process function `{}` registered'.format(key))


def get_post_forward_proc_func(key):
    """
    Gets a registered post-forward process function.

    :param key: unique key to identify the registered post-forward process function.
    :type key: str
    :return: registered post-forward process function.
    :rtype: typing.Callable
    """
    if key in POST_FORWARD_PROC_FUNC_DICT:
        return POST_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No post-forward process function `{}` registered'.format(key))


def get_post_epoch_proc_func(key):
    """
    Gets a registered post-epoch process function.

    :param key: unique key to identify the registered post-epoch process function.
    :type key: str
    :return: registered post-epoch process function.
    :rtype: typing.Callable
    """
    if key in POST_EPOCH_PROC_FUNC_DICT:
        return POST_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No post-epoch process function `{}` registered'.format(key))

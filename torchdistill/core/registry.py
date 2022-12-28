PRE_EPOCH_PROC_FUNC_DICT = dict()
PRE_FORWARD_PROC_FUNC_DICT = dict()
FORWARD_PROC_FUNC_DICT = dict()
POST_FORWARD_PROC_FUNC_DICT = dict()
POST_EPOCH_PROC_FUNC_DICT = dict()


def register_pre_epoch_proc_func(arg=None, **kwargs):
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
    if key in PRE_EPOCH_PROC_FUNC_DICT:
        return PRE_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No pre-epoch process function `{}` registered'.format(key))


def get_pre_forward_proc_func(key):
    if key in PRE_FORWARD_PROC_FUNC_DICT:
        return PRE_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No pre-forward process function `{}` registered'.format(key))


def get_forward_proc_func(key):
    if key is None:
        return FORWARD_PROC_FUNC_DICT['forward_batch_only']
    elif key in FORWARD_PROC_FUNC_DICT:
        return FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No forward process function `{}` registered'.format(key))


def get_post_forward_proc_func(key):
    if key in POST_FORWARD_PROC_FUNC_DICT:
        return POST_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No post-forward process function `{}` registered'.format(key))


def get_post_epoch_proc_func(key):
    if key in POST_EPOCH_PROC_FUNC_DICT:
        return POST_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No post-epoch process function `{}` registered'.format(key))

FORWARD_PROC_FUNC_DICT = dict()


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


def get_forward_proc_func(key):
    if key is None:
        return FORWARD_PROC_FUNC_DICT['forward_batch_only']
    elif key in FORWARD_PROC_FUNC_DICT:
        return FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No forward process function `{}` registered'.format(key))

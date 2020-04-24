CLASS_DICT = dict()
FUNC_DICT = dict()


def register_class(cls):
    CLASS_DICT[cls.__name__] = cls
    return cls


def register_func(func):
    FUNC_DICT[func.__name__] = func
    return func

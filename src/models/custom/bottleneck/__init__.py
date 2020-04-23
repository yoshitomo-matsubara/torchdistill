from models.custom.bottleneck import *

BOTTLENECK_CLASS_DICT = dict()
BOTTLENECK_FUNC_DICT = dict()


def register_bottleneck_class(cls):
    BOTTLENECK_CLASS_DICT[cls.__name__] = cls
    return cls


def register_bottleneck_func(func):
    BOTTLENECK_FUNC_DICT[func.__name__] = func
    return func

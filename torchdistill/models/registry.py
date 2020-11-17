MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()


def register_model_class(cls):
    MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_model_func(func):
    MODEL_FUNC_DICT[func.__name__] = func
    return func

import torch


MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()


def register_model_class(arg=None, **kwargs):
    def _register_model_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model_class(arg)
    return _register_model_class


def register_model_func(arg=None, **kwargs):
    def _register_model_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        MODEL_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_model_func(arg)
    return _register_model_func


def get_model(model_name, repo_or_dir=None, **kwargs):
    if model_name in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[model_name](**kwargs)
    elif model_name in MODEL_FUNC_DICT:
        return MODEL_FUNC_DICT[model_name](**kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, model_name, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(model_name))

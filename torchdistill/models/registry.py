import torch


MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()


def register_model_class(cls):
    MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_model_func(func):
    MODEL_FUNC_DICT[func.__name__] = func
    return func


def get_model(model_name, repo_or_dir=None, **kwargs):
    if model_name in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[model_name](**kwargs)
    elif model_name in MODEL_FUNC_DICT:
        return MODEL_FUNC_DICT[model_name](**kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, model_name, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(model_name))

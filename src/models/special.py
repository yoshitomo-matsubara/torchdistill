from torch import nn

CLASS_DICT = dict()


def register_special_module(cls):
    CLASS_DICT[cls.__name__] = cls
    return cls


@register_special_module
class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args


def get_adaptation_module(class_name, *args, **kwargs):
    if class_name not in CLASS_DICT:
        print('No special module called `{}` is registered.'.format(class_name))
        return None

    instance = CLASS_DICT[class_name](*args, **kwargs)
    return instance

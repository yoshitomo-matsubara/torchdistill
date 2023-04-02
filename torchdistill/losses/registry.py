from ..common import misc_util

LOSS_DICT = misc_util.get_classes_as_dict('torch.nn.modules.loss')
HIGH_LEVEL_LOSS_DICT = dict()
LOSS_WRAPPER_DICT = dict()
SINGLE_LOSS_DICT = dict()
FUNC2EXTRACT_MODEL_OUTPUT_DICT = dict()


def register_high_level_loss(arg=None, **kwargs):
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
    def _register_loss_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        LOSS_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_loss_wrapper(arg)
    return _register_loss_wrapper


def register_single_loss(arg=None, **kwargs):
    def _register_single_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        SINGLE_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_single_loss(arg)
    return _register_single_loss


def register_func2extract_model_output(arg=None, **kwargs):
    def _register_func2extract_model_output(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FUNC2EXTRACT_MODEL_OUTPUT_DICT[key] = func
        return func

    if callable(arg):
        return _register_func2extract_model_output(arg)
    return _register_func2extract_model_output


def get_loss(key, **kwargs):
    lower_loss_type = key.lower()
    if lower_loss_type in LOSS_DICT:
        return LOSS_DICT[lower_loss_type](**kwargs)
    raise ValueError('No loss `{}` registered'.format(key))


def get_high_level_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in HIGH_LEVEL_LOSS_DICT:
        return HIGH_LEVEL_LOSS_DICT[criterion_type](criterion_config)
    raise ValueError('No high-level loss `{}` registered'.format(criterion_type))


def get_loss_wrapper(single_loss, criterion_wrapper_config):
    wrapper_type = criterion_wrapper_config['type']
    if wrapper_type in LOSS_WRAPPER_DICT:
        return LOSS_WRAPPER_DICT[wrapper_type](single_loss, **criterion_wrapper_config.get('kwargs', dict()))
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_type))


def get_single_loss(single_criterion_config, criterion_wrapper_config=None):
    loss_type = single_criterion_config['type']
    single_loss = SINGLE_LOSS_DICT[loss_type](**single_criterion_config['kwargs']) \
        if loss_type in SINGLE_LOSS_DICT else get_loss(loss_type, **single_criterion_config['kwargs'])
    if criterion_wrapper_config is None:
        return single_loss
    return get_loss_wrapper(single_loss, criterion_wrapper_config)


def get_func2extract_model_output(key):
    if key is None:
        key = 'extract_simple_model_loss'
    if key in FUNC2EXTRACT_MODEL_OUTPUT_DICT:
        return FUNC2EXTRACT_MODEL_OUTPUT_DICT[key]
    raise ValueError('No function to extract original output `{}` registered'.format(key))

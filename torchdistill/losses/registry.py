from ..common import misc_util

LOSS_DICT = misc_util.get_classes_as_dict('torch.nn.modules.loss')
CUSTOM_LOSS_CLASS_DICT = dict()
LOSS_WRAPPER_CLASS_DICT = dict()
SINGLE_LOSS_CLASS_DICT = dict()
ORG_LOSS_LIST = list()
FUNC2EXTRACT_ORG_OUTPUT_DICT = dict()


def register_custom_loss(arg=None, **kwargs):
    def _register_custom_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        CUSTOM_LOSS_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_custom_loss(arg)
    return _register_custom_loss


def register_loss_wrapper(arg=None, **kwargs):
    def _register_loss_wrapper(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        LOSS_WRAPPER_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_loss_wrapper(arg)
    return _register_loss_wrapper


def register_single_loss(arg=None, **kwargs):
    def _register_single_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SINGLE_LOSS_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_single_loss(arg)
    return _register_single_loss


def register_org_loss(arg=None, **kwargs):
    def _register_org_loss(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SINGLE_LOSS_CLASS_DICT[key] = cls
        ORG_LOSS_LIST.append(cls)
        return cls

    if callable(arg):
        return _register_org_loss(arg)
    return _register_org_loss


def register_func2extract_org_output(arg=None, **kwargs):
    def _register_func2extract_org_output(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FUNC2EXTRACT_ORG_OUTPUT_DICT[key] = func
        return func

    if callable(arg):
        return _register_func2extract_org_output(arg)
    return _register_func2extract_org_output


def get_loss(loss_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()
    lower_loss_type = loss_type.lower()
    if lower_loss_type in LOSS_DICT:
        return LOSS_DICT[lower_loss_type](**param_dict, **kwargs)
    raise ValueError('No loss `{}` registered'.format(loss_type))


def get_custom_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in CUSTOM_LOSS_CLASS_DICT:
        return CUSTOM_LOSS_CLASS_DICT[criterion_type](criterion_config)
    raise ValueError('No custom loss `{}` registered'.format(criterion_type))


def get_loss_wrapper(single_loss, params_config, wrapper_config):
    wrapper_type = wrapper_config.get('type', None)
    if wrapper_type is None:
        return LOSS_WRAPPER_CLASS_DICT['SimpleLossWrapper'](single_loss, params_config)
    elif wrapper_type in LOSS_WRAPPER_CLASS_DICT:
        return LOSS_WRAPPER_CLASS_DICT[wrapper_type](single_loss, params_config, **wrapper_config.get('params', dict()))
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_type))


def get_single_loss(single_criterion_config, params_config=None):
    loss_type = single_criterion_config['type']
    single_loss = SINGLE_LOSS_CLASS_DICT[loss_type](**single_criterion_config['params']) \
        if loss_type in SINGLE_LOSS_CLASS_DICT else get_loss(loss_type, single_criterion_config['params'])
    if params_config is None:
        return single_loss
    return get_loss_wrapper(single_loss, params_config, params_config.get('wrapper', dict()))


def get_func2extract_org_output(func_name):
    if func_name is None:
        func_name = 'extract_simple_org_loss'
    if func_name in FUNC2EXTRACT_ORG_OUTPUT_DICT:
        return FUNC2EXTRACT_ORG_OUTPUT_DICT[func_name]
    raise ValueError('No function to extract original output `{}` registered'.format(func_name))

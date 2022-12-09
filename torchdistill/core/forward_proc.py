PROC_FUNC_DICT = dict()


def register_forward_proc_func(arg=None, **kwargs):
    def _register_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_forward_proc_func(arg)
    return _register_forward_proc_func


@register_forward_proc_func
def forward_batch_only(model, sample_batch, targets=None, supp_dict=None):
    return model(sample_batch)


@register_forward_proc_func
def forward_batch_target(model, sample_batch, targets, supp_dict=None):
    return model(sample_batch, targets)


@register_forward_proc_func
def forward_batch_supp_dict(model, sample_batch, targets, supp_dict=None):
    return model(sample_batch, supp_dict)


@register_forward_proc_func
def forward_batch4sskd(model, sample_batch, targets=None, supp_dict=None):
    c, h, w = sample_batch.size()[-3:]
    sample_batch = sample_batch.view(-1, c, h, w)
    return model(sample_batch)


def get_forward_proc_func(func_name):
    if func_name is None:
        return forward_batch_only
    elif func_name in PROC_FUNC_DICT:
        return PROC_FUNC_DICT[func_name]
    raise ValueError('No forward process function `{}` registered'.format(func_name))

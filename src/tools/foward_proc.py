PROC_FUNC_DICT = dict()


def register_func(func):
    PROC_FUNC_DICT[func.__name__] = func
    return func


@register_func
def forward_batch_only(model, sample_batch, targets=None, supp_dict=None):
    return model(sample_batch)


@register_func
def forward_batch_target(model, sample_batch, targets, supp_dict=None):
    return model(sample_batch, targets)


def get_forward_proc_func(func_name):
    if func_name not in PROC_FUNC_DICT:
        return forward_batch_only
    return PROC_FUNC_DICT[func_name]

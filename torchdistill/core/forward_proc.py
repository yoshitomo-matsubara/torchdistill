from .registry import register_forward_proc_func


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

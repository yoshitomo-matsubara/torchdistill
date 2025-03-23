from torchdistill.core.interfaces.forward_proc import register_forward_proc_func


@register_forward_proc_func
def forward_batch_as_kwargs(model, sample_batch, targets=None, supp_dict=None):
    return model(**sample_batch)

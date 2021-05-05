from torchdistill.core.forward_proc import register_forward_proc_func


@register_forward_proc_func
def forward_batch_as_kwargs(model, sample_batch, targets=None, supp_dict=None):
    return model(**sample_batch)


@register_forward_proc_func
def forward_batch_as_kwargs_for_distillbert(model, sample_batch, targets=None, supp_dict=None):
    inputs = {
        "input_ids": sample_batch[0],
        "attention_mask": sample_batch[1],
        "start_positions": sample_batch[3],
        "end_positions": sample_batch[4],
    }
    return model(**inputs)


@register_forward_proc_func
def forward_batch_as_kwargs_for_xlm(model, sample_batch, targets=None, supp_dict=None):
    inputs = {
        'input_ids': sample_batch[0],
        'attention_mask': sample_batch[1],
        'token_type_ids': None,
        'start_positions': sample_batch[3],
        'end_positions': sample_batch[4],
        'cls_index': sample_batch[5],
        'p_mask': sample_batch[6]
    }
    return model(**inputs)


@register_forward_proc_func
def forward_batch_as_kwargs_for_xlm_w_neg(model, sample_batch, targets=None, supp_dict=None):
    inputs = {
        'input_ids': sample_batch[0],
        'attention_mask': sample_batch[1],
        'token_type_ids': None,
        'start_positions': sample_batch[3],
        'end_positions': sample_batch[4],
        'cls_index': sample_batch[5],
        'p_mask': sample_batch[6],
        'is_impossible': sample_batch[7]
    }
    return model(**inputs)


@register_forward_proc_func
def forward_batch_as_kwargs_for_xlnet(model, sample_batch, targets=None, supp_dict=None):
    inputs = {
        'input_ids': sample_batch[0],
        'attention_mask': sample_batch[1],
        'token_type_ids': sample_batch[2],
        'start_positions': sample_batch[3],
        'end_positions': sample_batch[4],
        'cls_index': sample_batch[5],
        'p_mask': sample_batch[6]
    }
    return model(**inputs)


@register_forward_proc_func
def forward_batch_as_kwargs_for_xlnet_w_neg(model, sample_batch, targets=None, supp_dict=None):
    inputs = {
        'input_ids': sample_batch[0],
        'attention_mask': sample_batch[1],
        'token_type_ids': sample_batch[2],
        'start_positions': sample_batch[3],
        'end_positions': sample_batch[4],
        'cls_index': sample_batch[5],
        'p_mask': sample_batch[6],
        'is_impossible': sample_batch[7]
    }
    return model(**inputs)

import inspect
from functools import lru_cache

from torchdistill.common.module_util import unwrap_model
from torchdistill.core.interfaces.forward_proc import register_forward_proc_func


@lru_cache(maxsize=None)
def get_forward_param_keys(model_cls):
    """
    Returns the keyword argument names ``model_cls.forward`` accepts, or None if it accepts ``**kwargs``.

    The signature belongs to the class, not to the instance, so this is resolved once per architecture and
    reused for every batch.

    :param model_cls: model class (unwrapped, i.e. not a parallel wrapper).
    :type model_cls: type
    :return: accepted keyword argument names, or None if any keyword argument is accepted.
    :rtype: frozenset or None
    """
    params = inspect.signature(model_cls.forward).parameters
    if any(param.kind is param.VAR_KEYWORD for param in params.values()):
        return None
    return frozenset(params.keys())


@register_forward_proc_func
def forward_batch_as_kwargs(model, sample_batch, targets=None, supp_dict=None):
    """
    Feeds the batch to the model as keyword arguments.

    Entries the model does not accept are dropped, so that a batch carrying extra metadata (e.g. the prompt
    tensors the instruction pipeline adds for on-policy rollouts) can be fed to a model whose ``forward`` has
    an explicit signature. Models accepting ``**kwargs`` are given the batch as it is.
    """
    # A parallel wrapper's own forward takes **kwargs and would mask the wrapped model's signature
    param_keys = get_forward_param_keys(type(unwrap_model(model)))
    if param_keys is not None:
        sample_batch = {key: value for key, value in sample_batch.items() if key in param_keys}
    return model(**sample_batch)


@register_forward_proc_func
def forward_on_policy_sequences(model, sample_batch, targets=None, supp_dict=None):
    """
    Scores the on-policy sequences that ``generate_on_policy_sequences`` stored in ``sample_batch``.

    Used for the teacher, and for the student when the criterion has no model (task) loss term.
    """
    return model(
        input_ids=sample_batch['on_policy_input_ids'], attention_mask=sample_batch['on_policy_attention_mask']
    )


@register_forward_proc_func
def forward_on_policy_sequences_with_task_loss(model, sample_batch, targets=None, supp_dict=None):
    """
    Scores the on-policy sequences and additionally computes the off-policy task loss.

    The returned model output carries the on-policy logits for the KL term and the cross-entropy over the
    reference completions as ``loss``, which ``extract_transformers_loss`` turns into the model loss term
    (weighted by ``model_term`` in the criterion configuration). Use
    :func:`forward_on_policy_sequences` instead when the model loss term is not used, to skip this
    second forward pass.
    """
    model_outputs = forward_on_policy_sequences(model, sample_batch, targets, supp_dict)
    model_outputs.loss = model(
        input_ids=sample_batch['input_ids'], attention_mask=sample_batch['attention_mask'],
        labels=sample_batch['labels']
    ).loss
    return model_outputs

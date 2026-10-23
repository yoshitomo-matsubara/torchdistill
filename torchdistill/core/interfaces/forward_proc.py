import inspect
from functools import lru_cache

from .registry import register_forward_proc_func
from ...common.module_util import unwrap_model


@register_forward_proc_func
def forward_all(model, *args, **kwargs):
    """
    Performs forward computation using `*args` and `**kwargs`.

    :param model: model.
    :type model: nn.Module
    :param args: variable-length arguments for forward.
    :type args: tuple
    :param kwargs: kwargs for forward.
    :type kwargs: dict
    :return: model's forward output.
    :rtype: Any
    """
    return model(*args, **kwargs)


@register_forward_proc_func
def forward_batch_only(model, sample_batch, targets=None, supp_dict=None, **kwargs):
    """
    Performs forward computation using `sample_batch` only.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: Any
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    return model(sample_batch)


@register_forward_proc_func
def forward_batch_only_as_kwargs(model, sample_batch, targets=None, supp_dict=None):
    """
    Performs forward computation using `sample_batch` only.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: dict
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    return model(**sample_batch)


@register_forward_proc_func
def forward_batch_target(model, sample_batch, targets, supp_dict=None, **kwargs):
    """
    Performs forward computation using `sample_batch` and `targets` only.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: Any
    :param targets: training targets.
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    return model(sample_batch, targets)


@register_forward_proc_func
def forward_batch_supp_dict(model, sample_batch, targets, supp_dict=None, **kwargs):
    """
    Performs forward computation using `sample_batch` and `supp_dict` only.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: Any
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict.
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    return model(sample_batch, supp_dict)


@register_forward_proc_func
def forward_batch4sskd(model, sample_batch, targets=None, supp_dict=None, **kwargs):
    """
    Performs forward computation using `sample_batch` only for the SSKD method.

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: Any
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    c, h, w = sample_batch.size()[-3:]
    sample_batch = sample_batch.view(-1, c, h, w)
    return model(sample_batch)


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
    Performs forward computation using `sample_batch` as keyword arguments.

    Unlike :func:`forward_batch_only_as_kwargs`, entries the model does not accept are dropped, so that a batch
    carrying extra metadata (e.g., the prompt tensors for on-policy rollouts) can be fed to a model whose
    ``forward`` has an explicit signature. Models accepting ``**kwargs`` are given the batch as it is.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: dict
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    # A parallel wrapper's own forward takes **kwargs and would mask the wrapped model's signature
    param_keys = get_forward_param_keys(type(unwrap_model(model)))
    if param_keys is not None:
        sample_batch = {key: value for key, value in sample_batch.items() if key in param_keys}
    return model(**sample_batch)


@register_forward_proc_func
def forward_on_policy_sequences(model, sample_batch, targets=None, supp_dict=None):
    """
    Performs forward computation on the on-policy sequences that
    :func:`torchdistill.core.interfaces.pre_forward_proc.generate_on_policy_sequences` stored in `sample_batch`.

    Used for the teacher, and for the student when the criterion has no model (task) loss term.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: dict
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    attention_mask = sample_batch['on_policy_attention_mask']
    model_kwargs = {'input_ids': sample_batch['on_policy_input_ids'], 'attention_mask': attention_mask}
    # The prompts are left-padded, and without explicit position ids most models count positions from the first
    # (padding) column. `generate` derives them from the attention mask instead, so the scoring forward has to
    # match it. This matters for learned absolute positions (e.g. GPT-2), not for RoPE, and models without
    # positional ids (e.g. ALiBi in BLOOM) do not accept the argument at all
    param_keys = get_forward_param_keys(type(unwrap_model(model)))
    if param_keys is None or 'position_ids' in param_keys:
        model_kwargs['position_ids'] = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    return model(**model_kwargs)


@register_forward_proc_func
def forward_on_policy_sequences_with_task_loss(model, sample_batch, targets=None, supp_dict=None):
    """
    Performs forward computation on the on-policy sequences and additionally computes the off-policy task loss.

    The returned model output carries the on-policy logits for the KL term and the loss over the reference
    sequences (``input_ids``, ``attention_mask``, and ``labels`` in `sample_batch`) as ``loss``, which
    ``func2extract_model_loss`` in the criterion configuration turns into the model loss term. Use
    :func:`forward_on_policy_sequences` instead when the model loss term is not used, to skip this
    second forward pass.

    :param model: model.
    :type model: nn.Module
    :param sample_batch: sample batch.
    :type sample_batch: dict
    :param targets: training targets (won't be passed to forward).
    :type targets: Any
    :param supp_dict: supplementary dict (won't be passed to forward).
    :type supp_dict: dict
    :return: model's forward output.
    :rtype: Any
    """
    model_outputs = forward_on_policy_sequences(model, sample_batch, targets, supp_dict)
    model_outputs.loss = model(
        input_ids=sample_batch['input_ids'], attention_mask=sample_batch['attention_mask'],
        labels=sample_batch['labels']
    ).loss
    return model_outputs

from .registry import register_forward_proc_func


@register_forward_proc_func
def forward_default(model, *args, **kwargs):
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

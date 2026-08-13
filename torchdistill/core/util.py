import warnings
from collections import abc

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.scatter_gather import gather

from ..common.constant import def_logger
from ..common.module_util import get_module, check_if_wrapped
from ..common.constant import SELF_MODULE_PATH
from ..core.forward_hook import clear_io_dict_values

logger = def_logger.getChild(__name__)


def add_kwargs_to_io_dict(io_dict, module_path, **kwargs):
    """
    Adds kwargs to an I/O dict.

    .. deprecated:: 1.2.0
        Forward hooks initialize their own entries in an I/O dict, and this function is no longer used
        internally. It will be removed in a future release.

    :param io_dict: I/O dict.
    :type io_dict: dict
    :param module_path: module path.
    :type module_path: str
    :param kwargs: kwargs to be stored in ``io_dict``.
    :type kwargs: dict
    """
    warnings.warn(
        '`add_kwargs_to_io_dict` is deprecated and will be removed in a future release',
        DeprecationWarning, stacklevel=2
    )
    io_dict[module_path] = kwargs


def _extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def _resolve_module_path_flag(flag_config, target_module_path_set):
    """
    Resolves a forward hook flag that is given either as a bool (applied to all the target modules)
    or as a list of module paths (applied to the listed target modules only).

    :param flag_config: bool or list of module paths.
    :type flag_config: bool or list[str] or None
    :param target_module_path_set: set of all the target module paths.
    :type target_module_path_set: set[str]
    :return: set of module paths the flag is enabled for.
    :rtype: set[str]
    """
    if flag_config is None or flag_config is False:
        return set()
    if flag_config is True:
        return set(target_module_path_set)

    module_path_set = set(flag_config)
    unknown_module_path_set = module_path_set - target_module_path_set
    if len(unknown_module_path_set) > 0:
        raise ValueError(
            'module path(s) {} should be listed in `input` and/or `output` of the forward hook configuration'.format(
                sorted(unknown_module_path_set)
            )
        )
    return module_path_set


def set_hooks(model, unwrapped_org_model, model_config, forward_hook_manager):
    """
    Sets forward hooks for target modules in model, using ``forward_hook_manager``.

    ``model_config['forward_hook']`` accepts the following keys:

    * ``input``: list of module paths whose input should be stored.
    * ``output``: list of module paths whose output should be stored.
    * ``accumulates``: bool (applied to all the target modules) or list of module paths whose input/output
      should be accumulated across forward passes instead of being overwritten. Useful for autoregressive
      use cases such as on-policy distillation, where the target modules are called once per generated token.
    * ``stacks_accumulated``: bool or list of module paths whose accumulated per-step tensors should be
      stacked into a single tensor by :meth:`torchdistill.core.forward_hook.ForwardHookManager.pop_io_dict`.
      Requires the same module paths to be accumulated, and that the per-step tensors share the same shape.

    :param model: model.
    :type model: nn.Module
    :param unwrapped_org_model: unwrapped original model.
    :type unwrapped_org_model: nn.Module
    :param model_config: model configuration.
    :type model_config: dict
    :param forward_hook_manager: forward hook manager to register the forward hooks with.
    :type forward_hook_manager: torchdistill.core.forward_hook.ForwardHookManager
    :return: list of pairs of module path and removable forward hook handle.
    :rtype: list[(str, torch.utils.hook.RemovableHandle)]
    """
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list

    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    target_module_path_set = input_module_path_set.union(output_module_path_set)
    accumulating_module_path_set = \
        _resolve_module_path_flag(forward_hook_config.get('accumulates', None), target_module_path_set)
    stacking_module_path_set = \
        _resolve_module_path_flag(forward_hook_config.get('stacks_accumulated', None), target_module_path_set)
    if forward_hook_config.get('stacks_accumulated', None) is True:
        # `stacks_accumulated: True` should only apply to the accumulated module paths
        stacking_module_path_set &= accumulating_module_path_set

    for target_module_path in target_module_path_set:
        target_module = _extract_module(unwrapped_org_model, model, target_module_path)
        pair = forward_hook_manager.add_hook_to_module(
            target_module, target_module_path,
            requires_input=target_module_path in input_module_path_set,
            requires_output=target_module_path in output_module_path_set,
            accumulates=target_module_path in accumulating_module_path_set,
            stacks_accumulated=target_module_path in stacking_module_path_set
        )
        pair_list.append(pair)
    return pair_list


def wrap_model(
        model, model_config, device, device_ids=None, distributed=False,
        find_unused_parameters=False, any_updatable=True
):
    """
    Wraps ``model`` with DataParallel, DistributedDataParallel, FullyShardedDataParallel (FSDP), or
    FSDP2 (``fully_shard``) if specified.

    ``model_config['wrapper']['key']`` selects the wrapper and accepts one of
    'DataParallel', 'DistributedDataParallel', 'FullyShardedDataParallel' (FSDP), or
    'FullyShardedDataParallel2' (FSDP2). ``model_config['wrapper']['kwargs']`` are forwarded to the
    wrapper's constructor (e.g., ``auto_wrap_policy``, ``sharding_strategy``, ``mixed_precision`` for FSDP).

    .. note::
        FSDP/FSDP2 shard parameters across the process group, so ``model``'s ``state_dict()`` will only
        reflect the local shard. Use :func:`torchdistill.common.module_util.get_full_state_dict` /
        :func:`torchdistill.common.module_util.load_full_state_dict` for checkpointing instead of calling
        ``state_dict()`` / ``load_state_dict()`` directly.

    :param model: model.
    :type model: nn.Module
    :param model_config: model configuration.
    :type model_config: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool
    :param any_updatable: True if ``model`` contains any updatable parameters.
    :type any_updatable: bool
    :return: wrapped model (or ``model`` if wrapper is not specified).
    :rtype: nn.Module
    """
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    wrapper_kwargs = dict()
    if isinstance(wrapper, dict):
        wrapper_key = wrapper.get('key', None)
        wrapper_kwargs = wrapper.get('kwargs', wrapper_kwargs)
    else:
        wrapper_key = wrapper

    model.to(device)
    if wrapper_key is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper_key == 'FullyShardedDataParallel2' and distributed:
            model = fully_shard(model, **wrapper_kwargs)
        elif wrapper_key == 'FullyShardedDataParallel' and distributed:
            if device_ids:
                wrapper_kwargs.setdefault('device_id', device_ids[0])
            model = FullyShardedDataParallel(model, **wrapper_kwargs)
        elif wrapper_key == 'DistributedDataParallel' and distributed and any_updatable:
            wrapper_kwargs['device_ids'] = device_ids
            if 'find_unused_parameters' not in wrapper_kwargs:
                wrapper_kwargs['find_unused_parameters'] = find_unused_parameters

            model = DistributedDataParallel(model, **wrapper_kwargs)
        elif wrapper_key in {'DataParallel', 'DistributedDataParallel'}:
            wrapper_kwargs['device_ids'] = device_ids
            model = DataParallel(model, **wrapper_kwargs)
    return model


def clear_io_dict(model_io_dict):
    """
    Clears a model I/O dict's sub dict(s).

    Each module path is left with an empty dict, and the forward hooks repopulate the I/O type entries
    at the next forward pass.

    .. note::
        If you hold a :class:`torchdistill.core.forward_hook.ForwardHookManager`, prefer its
        :meth:`~torchdistill.core.forward_hook.ForwardHookManager.clear_io_dict` method. Both share
        :func:`torchdistill.core.forward_hook.clear_io_dict_values` as their implementation.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    """
    clear_io_dict_values(model_io_dict)


def extract_io_dict(model_io_dict, target_device):
    """
    Extracts I/O dict, gathering tensors on ``target_device``.

    .. deprecated:: 1.2.0
        Use :meth:`torchdistill.core.forward_hook.ForwardHookManager.pop_io_dict` instead, which additionally
        supports accumulated I/O. Unlike :meth:`~torchdistill.core.forward_hook.ForwardHookManager.pop_io_dict`,
        this function always adds a :obj:`torchdistill.common.constant.SELF_MODULE_PATH` entry, so replace
        ``io_dict[SELF_MODULE_PATH]['output'] = outputs`` with ``io_dict[SELF_MODULE_PATH] = {'output': outputs}``
        when migrating. This function will be removed in a future release.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    :param target_device: target device.
    :type target_device: torch.device or str
    :return: extracted I/O dict.
    :rtype: dict
    """
    warnings.warn(
        '`extract_io_dict` is deprecated and will be removed in a future release; '
        'use `ForwardHookManager.pop_io_dict` instead',
        DeprecationWarning, stacklevel=2
    )
    uses_cuda = target_device.type == 'cuda'
    gathered_io_dict = {SELF_MODULE_PATH: dict()}
    for module_path, module_io_dict in model_io_dict.items():
        gathered_io_dict[module_path] = dict()
        for io_type in list(module_io_dict.keys()):
            sub_dict = module_io_dict.pop(io_type)
            values = [sub_dict[key] for key in sorted(sub_dict.keys())]
            gathered_obj = gather(values, target_device) if uses_cuda and len(values) > 1 else values[-1]
            gathered_io_dict[module_path][io_type] = gathered_obj
    return gathered_io_dict


def update_io_dict(main_io_dict, sub_io_dict):
    """
    Updates an I/O dict with a sub I/O dict.

    :param main_io_dict: main I/O dict to be updated.
    :type main_io_dict: dict
    :param sub_io_dict: sub I/O dict.
    :type sub_io_dict: dict
    """
    for key, module_io_dict in sub_io_dict.items():
        for io_type, value in module_io_dict.items():
            # Tensors are always treated as stored values as `len` is not defined for 0-dim tensors and
            # would drop empty batches, while empty containers mean that nothing was stored
            if isinstance(value, torch.Tensor) or not isinstance(value, abc.Sized) or len(value) > 0:
                main_io_dict[key][io_type] = value


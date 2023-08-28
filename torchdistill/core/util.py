from collections import abc

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.scatter_gather import gather

from ..common.constant import def_logger
from ..common.module_util import get_module, check_if_wrapped
from ..common.constant import SELF_MODULE_PATH
from ..core.forward_hook import register_forward_hook_with_dict

logger = def_logger.getChild(__name__)


def add_kwargs_to_io_dict(io_dict, module_path, **kwargs):
    """
    Adds kwargs to an I/O dict.

    :param io_dict: I/O dict.
    :type io_dict: dict
    :param module_path: module path.
    :type module_path: str
    :param kwargs: kwargs to be stored in ``io_dict``.
    :type kwargs: dict
    """
    io_dict[module_path] = kwargs


def _extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def set_hooks(model, unwrapped_org_model, model_config, io_dict):
    """
    Sets forward hooks for target modules in model.

    :param model: model.
    :type model: nn.Module
    :param unwrapped_org_model: unwrapped original model.
    :type unwrapped_org_model: nn.Module
    :param model_config: model configuration.
    :type model_config: dict
    :param io_dict: I/O dict.
    :type io_dict: dict
    :return: list of pairs of module path and removable forward hook handle.
    :rtype: list[(str, torch.utils.hook.RemovableHandle)]
    """
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list

    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    for target_module_path in input_module_path_set.union(output_module_path_set):
        requires_input = target_module_path in input_module_path_set
        requires_output = target_module_path in output_module_path_set
        add_kwargs_to_io_dict(io_dict, target_module_path)
        target_module = _extract_module(unwrapped_org_model, model, target_module_path)
        handle = register_forward_hook_with_dict(target_module, target_module_path,
                                                 requires_input, requires_output, io_dict)
        pair_list.append((target_module_path, handle))
    return pair_list


def wrap_model(model, model_config, device, device_ids=None, distributed=False,
               find_unused_parameters=False, any_updatable=True):
    """
    Wraps ``model`` with either DataParallel or DistributedDataParallel if specified.

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
    model.to(device)
    if wrapper is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper == 'DistributedDataParallel' and distributed and any_updatable:
            model = DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
        elif wrapper in {'DataParallel', 'DistributedDataParallel'}:
            model = DataParallel(model, device_ids=device_ids)
    return model


def change_device(data, device):
    """
    Updates the device of tensor(s) stored in ``data``  with a new ``device``.

    :param data: data that contain tensor(s).
    :type data: Any
    :param device: new device.
    :type device: torch.device or str
    :return: ``data`` on the new ``device``.
    :rtype: Any
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(change_device(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(change_device(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: change_device(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [change_device(samples, device) for samples in transposed]
    return data


def tensor2numpy2tensor(data, device):
    """
    Converts tensor to numpy data and re-converts the numpy data to tensor.

    :param data: data that contain tensor(s).
    :type data: Any
    :param device: new device.
    :type device: torch.device or str
    :return: data that contain recreated tensor(s).
    :rtype: Any
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return torch.Tensor(data.to(device).data.numpy())
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(tensor2numpy2tensor(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(tensor2numpy2tensor(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: tensor2numpy2tensor(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [tensor2numpy2tensor(samples, device) for samples in transposed]
    return data


def clear_io_dict(model_io_dict):
    """
    Clears a model I/O dict's sub dict(s).

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    """
    for module_io_dict in model_io_dict.values():
        for sub_dict in list(module_io_dict.values()):
            sub_dict.clear()


def extract_io_dict(model_io_dict, target_device):
    """
    Extracts I/O dict, gathering tensors on ``target_device``.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    :param target_device: target device.
    :type target_device: torch.device or str
    :return: extracted I/O dict.
    :rtype: dict
    """
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
            if len(value) > 0:
                main_io_dict[key][io_type] = value


def extract_sub_model_io_dict(model_io_dict, index):
    """
    Extracts sub I/O dict from ``model_io_dict``.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    :param index: sample index.
    :type index: int
    :return: extracted sub I/O dict.
    :rtype: dict
    """
    sub_model_output_dict = dict()
    for module_path, sub_model_io_dict in model_io_dict.items():
        tmp_dict = dict()
        for key, value in sub_model_io_dict.items():
            tmp_dict[key] = value[index]
        sub_model_output_dict[module_path] = tmp_dict
    return sub_model_output_dict

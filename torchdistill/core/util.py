from collections import abc

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.scatter_gather import gather

from ..common.constant import def_logger
from ..common.module_util import get_module, check_if_wrapped
from ..core.forward_hook import register_forward_hook_with_dict

logger = def_logger.getChild(__name__)


def set_distillation_box_info(io_dict, module_path, **kwargs):
    io_dict[module_path] = kwargs


def extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def set_hooks(model, unwrapped_org_model, model_config, io_dict):
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list

    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    for target_module_path in input_module_path_set.union(output_module_path_set):
        requires_input = target_module_path in input_module_path_set
        requires_output = target_module_path in output_module_path_set
        set_distillation_box_info(io_dict, target_module_path)
        target_module = extract_module(unwrapped_org_model, model, target_module_path)
        handle = register_forward_hook_with_dict(target_module, target_module_path,
                                                 requires_input, requires_output, io_dict)
        pair_list.append((target_module_path, handle))
    return pair_list


def wrap_model(model, model_config, device, device_ids=None, distributed=False,
               find_unused_parameters=False, any_updatable=True):
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    model.to(device)
    if wrapper is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper == 'DistributedDataParallel' and distributed and any_updatable:
            model = DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
        elif wrapper in {'DataParallel', 'DistributedDataParallel'}:
            model = DataParallel(model, device_ids=device_ids)
    return model


def change_device(data, device):
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
    for module_io_dict in model_io_dict.values():
        for sub_dict in list(module_io_dict.values()):
            sub_dict.clear()


def extract_io_dict(model_io_dict, target_device):
    uses_cuda = target_device.type == 'cuda'
    gathered_io_dict = dict()
    for module_path, module_io_dict in model_io_dict.items():
        gathered_io_dict[module_path] = dict()
        for io_type in list(module_io_dict.keys()):
            sub_dict = module_io_dict.pop(io_type)
            values = [sub_dict[key] for key in sorted(sub_dict.keys())]
            gathered_obj = gather(values, target_device) if uses_cuda and len(values) > 1 else values[-1]
            gathered_io_dict[module_path][io_type] = gathered_obj
    return gathered_io_dict


def update_io_dict(main_io_dict, new_io_dict):
    for key, module_io_dict in new_io_dict.items():
        for io_type, value in module_io_dict.items():
            if len(value) > 0:
                main_io_dict[key][io_type] = value


def extract_sub_model_output_dict(model_output_dict, index):
    sub_model_output_dict = dict()
    for module_path, sub_model_io_dict in model_output_dict.items():
        tmp_dict = dict()
        for key, value in sub_model_io_dict.items():
            tmp_dict[key] = value[index]
        sub_model_output_dict[module_path] = tmp_dict
    return sub_model_output_dict

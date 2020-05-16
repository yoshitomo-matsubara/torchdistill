from collections import OrderedDict
from collections import abc

import torch
from torch.nn import Sequential, DataParallel
from torch.nn.parallel import DistributedDataParallel

from models.adaptation import get_adaptation_module
from myutils.pytorch.module_util import get_module, freeze_module_params, check_if_wrapped
from utils.constant import def_logger

logger = def_logger.getChild(__name__)


def redesign_model(org_model, model_config, model_label):
    logger.info('[{} model]'.format(model_label))
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the original {} model'.format(model_label))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

        for frozen_module_path in frozen_module_path_set:
            module = get_module(org_model, frozen_module_path)
            freeze_module_params(module)
        return org_model

    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())
    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['type'], **adaptation_config['params'])
        else:
            module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)
        module_dict[module_path.replace('.', '__attr__')] = module

    model = Sequential(module_dict)
    return model


def set_distillation_box_info(info_dict, module_path, **kwargs):
    info_dict[module_path] = kwargs


def extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def register_forward_hook_with_dict(module, module_path, requires_input, requires_output, info_dict):
    def forward_hook4input(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]
        info_dict[module_path]['input'] = func_input

    def forward_hook4output(self, func_input, func_output):
        info_dict[module_path]['output'] = func_output

    def forward_hook4io(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]
        info_dict[module_path]['input'] = func_input
        info_dict[module_path]['output'] = func_output

    if requires_input and not requires_output:
        return module.register_forward_hook(forward_hook4input)
    elif not requires_input and requires_output:
        return module.register_forward_hook(forward_hook4output)
    elif requires_input and requires_output:
        return module.register_forward_hook(forward_hook4io)
    raise ValueError('Either requires_input or requires_output should be True')


def set_hooks(model, unwrapped_org_model, model_config, info_dict):
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list

    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    for target_module_path in input_module_path_set.union(output_module_path_set):
        requires_input = target_module_path in input_module_path_set
        requires_output = target_module_path in output_module_path_set
        set_distillation_box_info(info_dict, target_module_path)
        target_module = extract_module(unwrapped_org_model, model, target_module_path)
        handle = register_forward_hook_with_dict(target_module, target_module_path,
                                                 requires_input, requires_output, info_dict)
        pair_list.append((target_module_path, handle))
    return pair_list


def wrap_model(model, model_config, device, device_ids=None, distributed=False):
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    model.to(device)
    if wrapper is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper == 'DistributedDataParallel' and distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
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


def extract_outputs(model_info_dict):
    model_output_dict = dict()
    for module_path, model_io_dict in model_info_dict.items():
        sub_model_io_dict = dict()
        for key in list(model_io_dict.keys()):
            sub_model_io_dict[key] = model_io_dict.pop(key)
        model_output_dict[module_path] = sub_model_io_dict
    return model_output_dict

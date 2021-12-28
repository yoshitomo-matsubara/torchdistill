from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn.parallel import DistributedDataParallel

from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import make_parent_dirs
from torchdistill.common.main_util import is_main_process, save_on_master
from torchdistill.common.module_util import check_if_wrapped, get_module, get_frozen_param_names, freeze_module_params
from torchdistill.models.adaptation import get_adaptation_module

logger = def_logger.getChild(__name__)


def wrap_if_distributed(model, device, device_ids, distributed):
    model.to(device)
    if distributed:
        any_frozen = len(get_frozen_param_names(model)) > 0
        return DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=any_frozen)
    return model


def load_module_ckpt(module, map_location, ckpt_file_path):
    state_dict = torch.load(ckpt_file_path, map_location=map_location)
    if check_if_wrapped(module):
        module.module.load_state_dict(state_dict)
    else:
        module.load_state_dict(state_dict)


def save_module_ckpt(module, ckpt_file_path):
    if is_main_process():
        make_parent_dirs(ckpt_file_path)
    state_dict = module.module.state_dict() if check_if_wrapped(module) else module.state_dict()
    save_on_master(state_dict, ckpt_file_path)


def add_submodule(module, module_path, module_dict):
    module_names = module_path.split('.')
    module_name = module_names.pop(0)
    if len(module_names) == 0:
        if module_name in module_dict:
            raise KeyError('module_name `{}` is already used.'.format(module_name))

        module_dict[module_name] = module
        return

    next_module_path = '.'.join(module_names)
    sub_module_dict = module_dict.get(module_name, None)
    if module_name not in module_dict:
        sub_module_dict = OrderedDict()
        module_dict[module_name] = sub_module_dict
    add_submodule(module, next_module_path, sub_module_dict)


def build_sequential_container(module_dict):
    for key in module_dict.keys():
        value = module_dict[key]
        if isinstance(value, OrderedDict):
            value = build_sequential_container(value)
            module_dict[key] = value
        elif not isinstance(value, Module):
            raise ValueError('module type `{}` is not expected'.format(type(value)))
    return Sequential(module_dict)


def redesign_model(org_model, model_config, model_label, model_type='original'):
    logger.info('[{} model]'.format(model_label))
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} {} model'.format(model_type, model_label))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

        isinstance_str = 'instance('
        for frozen_module_path in frozen_module_path_set:
            if frozen_module_path.startswith(isinstance_str) and frozen_module_path.endswith(')'):
                target_cls = nn.__dict__[frozen_module_path[len(isinstance_str):-1]]
                for m in org_model.modules():
                    if isinstance(m, target_cls):
                        freeze_module_params(m)
            else:
                module = get_module(org_model, frozen_module_path)
                freeze_module_params(module)
        return org_model

    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())

    for frozen_module_path in frozen_module_path_set:
        module = get_module(org_model, frozen_module_path)
        freeze_module_params(module)

    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['type'], **adaptation_config['params'])
        else:
            module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)
        add_submodule(module, module_path, module_dict)
    return build_sequential_container(module_dict)

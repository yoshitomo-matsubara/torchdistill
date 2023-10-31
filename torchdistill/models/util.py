from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn.parallel import DistributedDataParallel

from .registry import get_adaptation_module
from ..common.constant import def_logger
from ..common.file_util import make_parent_dirs
from ..common.main_util import is_main_process, save_on_master
from ..common.module_util import check_if_wrapped, get_module, get_frozen_param_names, get_updatable_param_names,\
    freeze_module_params

logger = def_logger.getChild(__name__)


def wrap_if_distributed(module, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
    """
    Wraps ``module`` with DistributedDataParallel if ``distributed`` = True and ``module`` has any updatable parameters.

    :param module: module to be wrapped.
    :type module: nn.Module
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    :return: wrapped module if ``distributed`` = True and it contains any updatable parameters.
    :rtype: nn.Module
    """
    module.to(device)
    if distributed and len(get_updatable_param_names(module)) > 0:
        any_frozen = len(get_frozen_param_names(module)) > 0
        if find_unused_parameters is None:
            find_unused_parameters = any_frozen
        return DistributedDataParallel(module, device_ids=device_ids, find_unused_parameters=find_unused_parameters,
                                       **kwargs)
    return module


def load_module_ckpt(module, map_location, ckpt_file_path):
    """
    Loads checkpoint for ``module``.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param map_location: ``map_location`` for torch.load.
    :type map_location: torch.device or str or dict or typing.Callable
    :param ckpt_file_path: file path to load checkpoint.
    :type ckpt_file_path: str
    """
    state_dict = torch.load(ckpt_file_path, map_location=map_location)
    if check_if_wrapped(module):
        module.module.load_state_dict(state_dict)
    else:
        module.load_state_dict(state_dict)


def save_module_ckpt(module, ckpt_file_path):
    """
    Saves checkpoint of ``module``'s state dict.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param ckpt_file_path: file path to save checkpoint.
    :type ckpt_file_path: str
    """
    if is_main_process():
        make_parent_dirs(ckpt_file_path)
    state_dict = module.module.state_dict() if check_if_wrapped(module) else module.state_dict()
    save_on_master(state_dict, ckpt_file_path)


def add_submodule(module, module_path, module_dict):
    """
    Recursively adds submodules to `module_dict`.

    :param module: module.
    :type module: nn.Module
    :param module_path: module path.
    :type module_path: str
    :param module_dict: module dict.
    :type module_dict: nn.ModuleDict or dict
    """
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
    """
    Builds sequential container (nn.Sequential) from ``module_dict``.

    :param module_dict: module dict to build sequential to build a sequential container.
    :type module_dict: nn.ModuleDict or collections.OrderedDict
    :return: sequential container.
    :rtype: nn.Sequential
    """
    for key in module_dict.keys():
        value = module_dict[key]
        if isinstance(value, OrderedDict):
            value = build_sequential_container(value)
            module_dict[key] = value
        elif not isinstance(value, Module):
            raise ValueError('module type `{}` is not expected'.format(type(value)))
    return Sequential(module_dict)


def redesign_model(org_model, model_config, model_label, model_type='original'):
    """
    Redesigns ``org_model`` and returns a new separate model e.g.,

    * prunes some modules from ``org_model``,
    * freezes parameters of some modules in ``org_model``, and
    * adds adaptation module(s) to ``org_model`` as a new separate model.

    .. note::
        The parameters and states of modules in ``org_model`` will be kept in a new redesigned model.

    :param org_model: original model to be redesigned.
    :type org_model: nn.Module
    :param model_config: configuration to redesign ``org_model``.
    :type model_config: dict
    :param model_label: model label (e.g., 'teacher', 'student') to be printed just for debugging purpose.
    :type model_label: str
    :param model_type: model type (e.g., 'original', name of model class, etc) to be printed just for debugging purpose.
    :type model_type: str
    :return: redesigned model.
    :rtype: nn.Module
    """
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} model'.format(model_type))
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
            module = get_adaptation_module(adaptation_config['key'], **adaptation_config['kwargs'])
        else:
            module = get_module(org_model, module_path)

        if module_path in frozen_module_path_set:
            freeze_module_params(module)

        add_submodule(module, module_path, module_dict)
    return build_sequential_container(module_dict)

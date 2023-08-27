from collections import OrderedDict

from torch.nn import DataParallel, Sequential, ModuleList, Module, Parameter
from torch.nn.parallel import DistributedDataParallel

from .constant import def_logger

logger = def_logger.getChild(__name__)


def check_if_wrapped(model):
    """
    Checks if a given model is wrapped by DataParallel or DistributedDataParallel.

    :param model: model.
    :type model: nn.Module
    :return: True if `model` is wrapped by either DataParallel or DistributedDataParallel.
    :rtype: bool
    """
    return isinstance(model, (DataParallel, DistributedDataParallel))


def count_params(module):
    """
    Returns the number of module parameters.

    :param module: module.
    :type module: nn.Module
    :return: number of model parameters.
    :rtype: int
    """
    return sum(param.numel() for param in module.parameters())


def freeze_module_params(module):
    """
    Freezes parameters by setting requires_grad=False for all the parameters.

    :param module: module.
    :type module: nn.Module
    """
    if isinstance(module, Module):
        for param in module.parameters():
            param.requires_grad = False
    elif isinstance(module, Parameter):
        module.requires_grad = False


def unfreeze_module_params(module):
    """
    Unfreezes parameters by setting requires_grad=True for all the parameters.

    :param module: module.
    :type module: nn.Module
    """
    if isinstance(module, Module):
        for param in module.parameters():
            param.requires_grad = True
    elif isinstance(module, Parameter):
        module.requires_grad = True


def get_updatable_param_names(module):
    """
    Gets collection of updatable parameter names.

    :param module: module.
    :type module: nn.Module
    :return: names of updatable parameters.
    :rtype: list[str]
    """
    return [name for name, param in module.named_parameters() if param.requires_grad]


def get_frozen_param_names(module):
    """
    Gets collection of frozen parameter names.

    :param module: module.
    :type module: nn.Module
    :return: names of frozen parameters.
    :rtype: list[str]
    """
    return [name for name, param in module.named_parameters() if not param.requires_grad]


def get_module(root_module, module_path):
    """
    Gets a module specified by ``module_path``.

    :param root_module: module.
    :type root_module: nn.Module
    :param module_path: module path for extracting the module from ``root_module``.
    :type module_path: str
    :return: module extracted from ``root_module`` if exists.
    :rtype: nn.Module or None
    """
    module_names = module_path.split('.')
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (DataParallel, DistributedDataParallel)):
                module = module.module
                if not hasattr(module, module_name):
                    if isinstance(module, Sequential) and module_name.lstrip('-').isnumeric():
                        module = module[int(module_name)]
                    else:
                        logger.info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
                                                                                       type(root_module).__name__))
                else:
                    module = getattr(module, module_name)
            elif isinstance(module, (Sequential, ModuleList)) and module_name.lstrip('-').isnumeric():
                module = module[int(module_name)]
            else:
                logger.info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
                                                                               type(root_module).__name__))
                return None
        else:
            module = getattr(module, module_name)
    return module


def get_hierarchized_dict(module_paths):
    """
    Gets a hierarchical structure from module paths.

    :param module_paths: module paths.
    :type module_paths: list[str]
    :return: module extracted from ``root_module`` if exists.
    :rtype: dict
    """
    children_dict = OrderedDict()
    for module_path in module_paths:
        elements = module_path.split('.')
        if elements[0] not in children_dict and len(elements) == 1:
            children_dict[elements[0]] = module_path
            continue
        elif elements[0] not in children_dict:
            children_dict[elements[0]] = list()
        children_dict[elements[0]].append('.'.join(elements[1:]))

    for key in children_dict.keys():
        value = children_dict[key]
        if isinstance(value, list) and len(value) > 1:
            children_dict[key] = get_hierarchized_dict(value)
    return children_dict


def decompose(ordered_dict):
    """
    Converts an ordered dict into a list of key-value pairs.

    :param ordered_dict: ordered dict.
    :type ordered_dict: collections.OrderedDict
    :return: list of key-value pairs.
    :rtype: list[(str, Any)]
    """
    component_list = list()
    for key, value in ordered_dict.items():
        if isinstance(value, OrderedDict):
            component_list.append((key, decompose(value)))
        elif isinstance(value, list):
            component_list.append((key, value))
        else:
            component_list.append(key)
    return component_list


def get_components(module_paths):
    """
    Converts module paths into a list of pairs of parent module and child module names.

    :param module_paths: module paths.
    :type module_paths: list[str]
    :return: list of pairs of parent module and child module names.
    :rtype: list[(str, str)]
    """
    ordered_dict = get_hierarchized_dict(module_paths)
    return decompose(ordered_dict)


def extract_target_modules(parent_module, target_class, module_list):
    """
    Extracts modules that are instance of ``target_class`` and update ``module_list`` with the extracted modules.

    :param parent_module: parent module.
    :type parent_module: nn.Module
    :param target_class: target class.
    :type target_class: class
    :param module_list: (empty) list to be filled with modules that are instances of ``target_class``.
    :type module_list: list[nn.Module]
    """
    if isinstance(parent_module, target_class):
        module_list.append(parent_module)

    child_modules = list(parent_module.children())
    for child_module in child_modules:
        extract_target_modules(child_module, target_class, module_list)


def extract_all_child_modules(parent_module, module_list):
    """
    Extracts all the child modules and update ``module_list`` with the extracted modules.

    :param parent_module: parent module.
    :type parent_module: nn.Module
    :param module_list: (empty) list to be filled with child modules.
    :type module_list: list[nn.Module]
    """
    child_modules = list(parent_module.children())
    if not child_modules:
        module_list.append(parent_module)
        return

    for child_module in child_modules:
        extract_all_child_modules(child_module, module_list)

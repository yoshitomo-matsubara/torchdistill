from collections import OrderedDict

from torch.nn import DataParallel, Sequential, ModuleList
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)


def check_if_wrapped(model):
    return isinstance(model, (DataParallel, DistributedDataParallel))


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def freeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = True


def get_updatable_param_names(module):
    return [name for name, param in module.named_parameters() if param.requires_grad]


def get_frozen_param_names(module):
    return [name for name, param in module.named_parameters() if not param.requires_grad]


def get_module(root_module, module_path):
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
    ordered_dict = get_hierarchized_dict(module_paths)
    return decompose(ordered_dict)


def extract_target_modules(parent_module, target_class, module_list):
    if isinstance(parent_module, target_class):
        module_list.append(parent_module)

    child_modules = list(parent_module.children())
    for child_module in child_modules:
        extract_target_modules(child_module, target_class, module_list)


def extract_all_child_modules(parent_module, module_list):
    child_modules = list(parent_module.children())
    if not child_modules:
        module_list.append(parent_module)
        return

    for child_module in child_modules:
        extract_all_child_modules(child_module, module_list)

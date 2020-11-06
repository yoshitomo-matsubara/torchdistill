from collections import OrderedDict

import torch
from torch.nn import DataParallel, Sequential
from torch.nn.parallel import DistributedDataParallel


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
                    if isinstance(module, Sequential):
                        module = module[int(module_name)]
                    else:
                        print('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
                                                                                 type(root_module).__name__))
                else:
                    module = getattr(module, module_name)
            elif isinstance(module, Sequential):
                module = module[int(module_name)]
            else:
                print('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
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


def extract_decomposable_modules(parent_module, z, module_list, output_size_list=None, first=True, exception_size=-1):
    parent_module.eval()
    child_modules = list(parent_module.children())
    if first:
        output_size_list = list()

    if not child_modules:
        module_list.append(parent_module)
        try:
            z = parent_module(z)
            output_size_list.append([*z.size()])
            return z, True
        except (RuntimeError, ValueError):
            try:
                z = parent_module(z.view(z.size(0), exception_size))
                output_size_list.append([*z.size()])
                return z, True
            except RuntimeError:
                ValueError('Error w/o child modules\t', type(parent_module).__name__)
        return z, False

    try:
        expected_z = parent_module(z)
    except (RuntimeError, ValueError):
        try:
            resized_z = z.view(z.size(0), exception_size)
            expected_z = parent_module(resized_z)
            z = resized_z
        except RuntimeError:
            ValueError('Error w/ child modules\t', type(parent_module).__name__)
            return z, False

    submodule_list = list()
    sub_output_size_list = list()
    decomposable = True
    for child_module in child_modules:
        z, decomposable = extract_decomposable_modules(child_module, z, submodule_list, sub_output_size_list, False)
        if not decomposable:
            break

    is_tensor = isinstance(expected_z, torch.Tensor) and isinstance(z, torch.Tensor)
    if decomposable and is_tensor and expected_z.size() == z.size() and expected_z.isclose(z).all().item() == 1:
        module_list.extend(submodule_list)
        output_size_list.extend(sub_output_size_list)
        return expected_z, True

    if decomposable and not is_tensor and type(expected_z) == type(z) and expected_z == z:
        module_list.extend(submodule_list)
        output_size_list.extend(sub_output_size_list)
    elif not first:
        module_list.append(parent_module)
        if is_tensor:
            output_size_list.append([*expected_z.size()])
        else:
            output_size_list.append(len(expected_z))
    elif not check_if_wrapped(parent_module) and len(module_list) == len(output_size_list) == 0\
            and len(submodule_list) > 0 and len(sub_output_size_list) > 0:
        module_list.extend(submodule_list)
        output_size_list.extend(sub_output_size_list)
    return expected_z, True


def extract_intermediate_io(x, module, module_paths):
    io_dict = OrderedDict()

    def forward_hook(self, input, output):
        path = self.__dict__['module_path']
        if path not in io_dict:
            io_dict[path] = list()
        io_dict[path].append((input, output))

    hook_list = list()
    for module_path in module_paths:
        target_module = get_module(module, module_path)
        target_module.__dict__['module_path'] = module_path
        hook = target_module.register_forward_hook(forward_hook)
        hook_list.append(hook)

    module(x)
    while len(hook_list) > 0:
        hook = hook_list.pop()
        hook.remove()
    return io_dict

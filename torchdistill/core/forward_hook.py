from collections import abc

import torch
from torch.nn.parallel.scatter_gather import gather

from ..common.module_util import check_if_wrapped, get_module


def get_device_index(data):
    """
    Gets device index of tensor in given data.

    :param data: tensor or data structure containing tensor.
    :type data: torch.Tensor or abc.Mapping or tuple or list
    :return: device index.
    :rtype: int or str or None
    """
    if isinstance(data, torch.Tensor):
        device = data.device
        return 'cpu' if device.type == 'cpu' else device.index
    elif isinstance(data, abc.Mapping):
        for key, data in data.items():
            result = get_device_index(data)
            if result is not None:
                return result
    elif isinstance(data, tuple):
        for d in data:
            result = get_device_index(d)
            if result is not None:
                return result
    elif isinstance(data, abc.Sequence) and not isinstance(data, (list, tuple)):
        for d in data:
            result = get_device_index(d)
            if result is not None:
                return result
    return None


def register_forward_hook_with_dict(root_module, module_path, requires_input, requires_output, io_dict):
    """
    Registers a forward hook for a child module to store its input and/or output in `io_dict`.

    :param root_module: root module (e.g., model).
    :type root_module: nn.Module
    :param module_path: path to target child module.
    :type module_path: str
    :param requires_input: if True, stores input to the target child module.
    :type requires_input: bool
    :param requires_output: if True, stores output from the target child module.
    :type requires_output: bool
    :param io_dict: dict to store the target child module's input and/or output.
    :type io_dict: dict
    :return: removable forward hook handle.
    :rtype: torch.utils.hook.RemovableHandle
    """
    io_dict[module_path] = dict()

    def forward_hook4input(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]

        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'input' not in sub_io_dict:
            sub_io_dict['input'] = dict()
        sub_io_dict['input'][device_index] = func_input

    def forward_hook4output(self, func_input, func_output):
        if isinstance(func_output, tuple) and len(func_output) == 1:
            func_output = func_output[0]

        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'output' not in sub_io_dict:
            sub_io_dict['output'] = dict()
        sub_io_dict['output'][device_index] = func_output

    def forward_hook4io(self, func_input, func_output):
        if isinstance(func_input, tuple) and len(func_input) == 1:
            func_input = func_input[0]
        if isinstance(func_output, tuple) and len(func_output) == 1:
            func_output = func_output[0]

        device_index = get_device_index(func_output)
        sub_io_dict = io_dict[module_path]
        if 'input' not in sub_io_dict:
            sub_io_dict['input'] = dict()

        if 'output' not in sub_io_dict:
            sub_io_dict['output'] = dict()

        sub_io_dict['input'][device_index] = func_input
        sub_io_dict['output'][device_index] = func_output

    if requires_input and not requires_output:
        return root_module.register_forward_hook(forward_hook4input)
    elif not requires_input and requires_output:
        return root_module.register_forward_hook(forward_hook4output)
    elif requires_input and requires_output:
        return root_module.register_forward_hook(forward_hook4io)
    raise ValueError('Either requires_input or requires_output should be True')


class ForwardHookManager(object):
    """
    A forward hook manager for PyTorch modules.

    :param target_device: target device.
    :type target_device: torch.device or str

    Example::
        >>> import torch
        >>> from torchvision import models
        >>> from torchdistill.core.forward_hook import ForwardHookManager
        >>> device = torch.device('cpu')
        >>> forward_hook_manager = ForwardHookManager(device)
        >>> model = models.resnet18()
        >>> forward_hook_manager.add_hook(model, 'layer2')
        >>> x = torch.rand(16, 3, 224, 224)
        >>> y = model(x)
        >>> io_dict = forward_hook_manager.pop_io_dict()
        >>> layer2_input_tensor = io_dict['layer2']['input']
        >>> layer2_output_tensor = io_dict['layer2']['output']
    """
    def __init__(self, target_device):
        self.target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        self.uses_cuda = self.target_device.type == 'cuda'
        self.io_dict = dict()
        self.hook_list = list()

    def add_hook(self, root_module, module_path, requires_input=True, requires_output=True):
        """
        Registers a forward hook for a child module to store its input and/or output.

        :param root_module: root module (e.g., model).
        :type root_module: nn.Module
        :param module_path: path to target child module.
        :type module_path: str
        :param requires_input: if True, stores input to the target child module.
        :type requires_input: bool
        :param requires_output: if True, stores output from the target child module.
        :type requires_output: bool
        """
        unwrapped_module = root_module.module if check_if_wrapped(root_module) else root_module
        sub_module = get_module(unwrapped_module, module_path)
        handle = \
            register_forward_hook_with_dict(sub_module, module_path, requires_input, requires_output, self.io_dict)
        self.hook_list.append((module_path, handle))

    def pop_io_dict(self):
        """
        Pops I/O dict after gathering tensors on ``self.target_device``.

        :return: I/O dict that contains input and/or output tensors with a module path as a key.
        :rtype: dict
        """
        gathered_io_dict = dict()
        for module_path, module_io_dict in self.io_dict.items():
            gathered_io_dict[module_path] = dict()
            for io_type in list(module_io_dict.keys()):
                sub_dict = module_io_dict.pop(io_type)
                values = [sub_dict[key] for key in sorted(sub_dict.keys())]
                gathered_obj = gather(values, self.target_device) if self.uses_cuda and len(values) > 1 else values[-1]
                gathered_io_dict[module_path][io_type] = gathered_obj
        return gathered_io_dict

    def pop_io_dict_from_device(self, device):
        """
        Pops I/O dict for a specified ``device``.

        :param device: device to pop I/O dict.
        :type device: torch.device
        :return: I/O dict that contains input and/or output tensors with a module path as a key.
        :rtype: dict
        """
        device_io_dict = dict()
        device_key = device.index if device.type == 'cuda' else device.type
        for module_path, module_io_dict in self.io_dict.items():
            device_io_dict[module_path] = dict()
            for io_type in list(module_io_dict.keys()):
                sub_dict = module_io_dict[io_type]
                device_io_dict[module_path][io_type] = sub_dict.pop(device_key)
        return device_io_dict

    def change_target_device(self, target_device):
        """
        Updates the target device with a new ``target_device``.

        :param target_device: new target device.
        :type target_device: torch.device or str
        """
        target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        if self.target_device.type != target_device.type:
            for sub_dict in self.io_dict.values():
                sub_dict.clear()
        self.target_device = target_device

    def clear(self):
        """
        Clears I/O dict and forward hooks registered in the instance.
        """
        self.io_dict.clear()
        for _, handle in self.hook_list:
            handle.remove()
        self.hook_list.clear()

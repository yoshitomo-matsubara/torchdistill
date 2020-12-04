from unittest import TestCase

import torch
from torchvision import models

from torchdistill.core.forward_hook import ForwardHookManager


class ForwardHookManagerUnitTest(TestCase):
    def test_init(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        assert fhm.target_device == device

    def test_add_hook(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'layer2'
        fhm.add_hook(model, target_module_path)
        assert fhm.hook_list[0][0] == target_module_path

    def test_pop_io_dict(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, requires_input=False, requires_output=True)
        x = torch.rand(1, 3, 224, 224)
        y = model(x)
        io_dict = fhm.pop_io_dict()
        assert len(io_dict) == 1
        assert 'output' in io_dict[target_module_path]
        hooked_y = io_dict[target_module_path]['output']
        assert torch.equal(y, hooked_y)
        assert len(fhm.io_dict[target_module_path]) == 0

    def test_pop_io_dict_from_device(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, requires_input=False, requires_output=True)
        x = torch.rand(1, 3, 224, 224)
        y = model(x)
        io_dict = fhm.pop_io_dict_from_device(device)
        assert len(io_dict) == 1
        assert 'output' in io_dict[target_module_path]
        hooked_y = io_dict[target_module_path]['output']
        assert torch.equal(y, hooked_y)
        assert len(fhm.io_dict[target_module_path]['output']) == 0

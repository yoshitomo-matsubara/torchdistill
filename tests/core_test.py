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

    def test_pop_io_dict_accumulates_false(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, requires_input=False, requires_output=True, accumulates=False)
        assert target_module_path not in fhm._accumulating_module_paths
        model(torch.rand(1, 3, 224, 224))
        y2 = model(torch.rand(1, 3, 224, 224))
        io_dict = fhm.pop_io_dict()
        hooked_output = io_dict[target_module_path]['output']
        assert isinstance(hooked_output, torch.Tensor)
        assert torch.equal(y2, hooked_output)

    def test_pop_io_dict_accumulates_true(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, requires_input=False, requires_output=True, accumulates=True)
        assert target_module_path in fhm._accumulating_module_paths
        num_steps = 3
        expected_outputs = [model(torch.rand(1, 3, 224, 224)) for _ in range(num_steps)]
        io_dict = fhm.pop_io_dict()
        hooked_outputs = io_dict[target_module_path]['output']
        assert isinstance(hooked_outputs, list)
        assert len(hooked_outputs) == num_steps
        for expected, actual in zip(expected_outputs, hooked_outputs):
            assert torch.equal(expected, actual)

    def test_clear_with_accumulates(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(False)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, accumulates=True)
        assert target_module_path in fhm._accumulating_module_paths
        fhm.clear()
        assert len(fhm._accumulating_module_paths) == 0
        assert len(fhm.hook_list) == 0
        assert len(fhm.io_dict) == 0

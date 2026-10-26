from types import SimpleNamespace
from unittest import TestCase

import torch
from torch import nn
from torch.nn import DataParallel
from torchvision import models

import torchdistill.core.interfaces.forward_proc  # noqa: registers forward process functions
import torchdistill.core.interfaces.post_epoch_proc  # noqa: registers post-epoch process functions
import torchdistill.core.interfaces.post_forward_proc  # noqa: registers post-forward process functions
import torchdistill.core.interfaces.pre_epoch_proc  # noqa: registers pre-epoch process functions
import torchdistill.core.interfaces.pre_forward_proc  # noqa: registers pre-forward process functions
import torchdistill.losses.high_level  # noqa: registers high-level losses
from torchdistill.common.constant import SELF_MODULE_PATH
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.forward_hook import ForwardHookManager
from torchdistill.core.interfaces.registry import get_forward_proc_func, get_pre_forward_proc_func
from torchdistill.core.training import get_training_box
from torchdistill.losses.registry import register_high_level_loss
from torchdistill.models.registry import register_auxiliary_model_wrapper
from torchdistill.models.wrapper import AuxiliaryModelWrapper


class ForwardHookManagerUnitTest(TestCase):
    def test_init(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        assert fhm.target_device == device

    def test_add_hook(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        target_module_path = 'layer2'
        fhm.add_hook(model, target_module_path)
        assert fhm.hook_list[0][0] == target_module_path

    def test_pop_io_dict(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
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
        model = models.resnet18(weights=None)
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
        model = models.resnet18(weights=None)
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
        model = models.resnet18(weights=None)
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

    def test_pop_io_dict_stacks_accumulated(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        target_module_path = 'fc'
        fhm.add_hook(
            model, target_module_path, requires_input=False, requires_output=True,
            accumulates=True, stacks_accumulated=True
        )
        assert target_module_path in fhm._accumulating_module_paths
        assert target_module_path in fhm._stacking_module_paths
        num_steps = 3
        expected_outputs = [model(torch.rand(1, 3, 224, 224)) for _ in range(num_steps)]
        io_dict = fhm.pop_io_dict()
        hooked_outputs = io_dict[target_module_path]['output']
        assert isinstance(hooked_outputs, torch.Tensor)
        assert hooked_outputs.shape == (num_steps, *expected_outputs[0].shape)
        for i, expected in enumerate(expected_outputs):
            assert torch.equal(expected, hooked_outputs[i])

    def test_add_hook_stacks_accumulated_without_accumulates_raises(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        with self.assertRaises(ValueError):
            fhm.add_hook(model, 'fc', accumulates=False, stacks_accumulated=True)

    def test_clear_io_dict_empties_module_entries_and_keeps_hooks(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, requires_input=False, requires_output=True)
        model(torch.rand(1, 3, 224, 224))
        assert len(fhm.io_dict[target_module_path]['output']) > 0
        fhm.clear_io_dict()
        # Each module path is left with an empty dict rather than empty I/O type entries
        assert fhm.io_dict[target_module_path] == dict()
        assert len(fhm.hook_list) == 1
        # The hooks are still registered, so the next forward pass repopulates the I/O dict
        y = model(torch.rand(1, 3, 224, 224))
        assert torch.equal(fhm.pop_io_dict()[target_module_path]['output'], y)

    def test_pop_io_dict_after_clear_io_dict(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        fhm.add_hook(model, 'fc', requires_input=False, requires_output=True)
        model(torch.rand(1, 3, 224, 224))
        fhm.clear_io_dict()
        # Popping without an intervening forward pass yields an empty entry instead of raising
        assert fhm.pop_io_dict() == {'fc': dict()}

    def test_change_target_device_updates_uses_cuda(self):
        fhm = ForwardHookManager(torch.device('cpu'))
        assert not fhm.uses_cuda
        fhm.change_target_device(torch.device('cuda:0'))
        assert fhm.target_device == torch.device('cuda:0')
        assert fhm.uses_cuda
        fhm.change_target_device('cpu')
        assert fhm.target_device == torch.device('cpu')
        assert not fhm.uses_cuda

    def test_change_target_device_clears_io_dict_across_device_types(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        fhm.add_hook(model, 'fc', requires_input=False, requires_output=True)
        model(torch.rand(1, 3, 224, 224))
        fhm.change_target_device(torch.device('cuda:0'))
        assert fhm.io_dict['fc'] == dict()
        assert len(fhm.hook_list) == 1

    def test_change_target_device_keeps_io_dict_within_same_device_type(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        fhm.add_hook(model, 'fc', requires_input=False, requires_output=True)
        y = model(torch.rand(1, 3, 224, 224))
        fhm.change_target_device('cpu')
        assert torch.equal(fhm.pop_io_dict()['fc']['output'], y)

    def test_clear_with_accumulates(self):
        device = torch.device('cpu')
        fhm = ForwardHookManager(device)
        model = models.resnet18(weights=None)
        target_module_path = 'fc'
        fhm.add_hook(model, target_module_path, accumulates=True, stacks_accumulated=True)
        assert target_module_path in fhm._accumulating_module_paths
        assert target_module_path in fhm._stacking_module_paths
        fhm.clear()
        assert len(fhm._accumulating_module_paths) == 0
        assert len(fhm._stacking_module_paths) == 0
        assert len(fhm.hook_list) == 0
        assert len(fhm.io_dict) == 0


class ToyNet(nn.Module):
    """Tiny model with named submodules to be targeted by forward hooks."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(8, 6), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(6, 4), nn.ReLU())
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        return self.fc(z2)


@register_high_level_loss(key='IoDictCaptureLoss')
class IoDictCaptureLoss(nn.Module):
    """High-level loss that captures the I/O dict given to it so that tests can inspect it."""
    def __init__(self):
        super().__init__()
        self.captured_io_dict = None
        self.captured_model_loss_dict = None

    def forward(self, io_dict, model_loss_dict, targets):
        self.captured_io_dict = io_dict
        self.captured_model_loss_dict = model_loss_dict
        # `extract_model_loss_dict` returns an empty dict unless the model output is a dict,
        # so always return a tensor rather than the int 0 that sum() would give for no terms
        sub_losses = [loss.mean() for loss in model_loss_dict.values()]
        if len(sub_losses) == 0:
            return torch.zeros((), requires_grad=True)
        return torch.stack(sub_losses).sum()


def build_train_config(model_config=None, teacher_model_config=None, student_model_config=None):
    train_config = {
        'num_epochs': 1,
        'criterion': {
            'key': 'IoDictCaptureLoss',
            'kwargs': dict()
        }
    }
    if model_config is not None:
        train_config['model'] = model_config
    if teacher_model_config is not None:
        train_config['teacher'] = teacher_model_config
    if student_model_config is not None:
        train_config['student'] = student_model_config
    return train_config


class TrainingBoxForwardHookUnitTest(TestCase):
    @staticmethod
    def build_box(forward_hook_config):
        model = ToyNet()
        model.eval()
        train_config = build_train_config(model_config={'forward_hook': forward_hook_config, 'forward_proc': 'forward_batch_only'})
        training_box = get_training_box(
            model, dict(), train_config, torch.device('cpu'), None, False, 1.0
        )
        return model, training_box

    def test_set_hooks_registers_configured_module_paths(self):
        model, training_box = self.build_box({'input': ['layer2'], 'output': ['layer1', 'fc']})
        hooked_module_paths = {module_path for module_path, _ in training_box.target_model_pairs}
        assert hooked_module_paths == {'layer1', 'layer2', 'fc'}
        assert set(training_box.model_io_dict.keys()) == {'layer1', 'layer2', 'fc'}
        training_box.clean_modules()

    def test_no_forward_hook_config_registers_nothing(self):
        model, training_box = self.build_box(dict())
        assert len(training_box.target_model_pairs) == 0
        assert len(training_box.model_io_dict) == 0
        training_box.clean_modules()

    def test_extracts_output_of_target_modules(self):
        model, training_box = self.build_box({'output': ['layer1', 'layer2', 'fc']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        # Expected values are computed after the forward pass so that these reference calls,
        # which also trigger the registered hooks, cannot make the assertions below pass spuriously
        expected_z1 = model.layer1(x)
        expected_z2 = model.layer2(expected_z1)
        expected_y = model.fc(expected_z2)
        assert torch.equal(student_io_dict['layer1']['output'], expected_z1)
        assert torch.equal(student_io_dict['layer2']['output'], expected_z2)
        assert torch.equal(student_io_dict['fc']['output'], expected_y)
        training_box.clean_modules()

    def test_extracts_input_of_target_modules(self):
        model, training_box = self.build_box({'input': ['layer2', 'fc']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_z1 = model.layer1(x)
        expected_z2 = model.layer2(expected_z1)
        # Input of layer2 is the output of layer1, and input of fc is the output of layer2
        assert torch.equal(student_io_dict['layer2']['input'], expected_z1)
        assert torch.equal(student_io_dict['fc']['input'], expected_z2)
        assert 'output' not in student_io_dict['layer2']
        assert 'output' not in student_io_dict['fc']
        training_box.clean_modules()

    def test_extracts_both_input_and_output_of_target_module(self):
        model, training_box = self.build_box({'input': ['fc'], 'output': ['fc']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        fc_io_dict = training_box.criterion.captured_io_dict['student']['fc']
        expected_z2 = model.layer2(model.layer1(x))
        expected_y = model.fc(expected_z2)
        assert set(fc_io_dict.keys()) == {'input', 'output'}
        assert torch.equal(fc_io_dict['input'], expected_z2)
        assert torch.equal(fc_io_dict['output'], expected_y)
        training_box.clean_modules()

    def test_io_dict_contains_model_output_at_self_module_path(self):
        model, training_box = self.build_box({'output': ['fc']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        io_dict = training_box.criterion.captured_io_dict
        assert torch.equal(io_dict['student'][SELF_MODULE_PATH]['output'], model(x))
        assert len(io_dict['teacher']) == 0
        training_box.clean_modules()

    def test_io_dict_is_refreshed_at_each_forward_process(self):
        model, training_box = self.build_box({'output': ['fc']})
        training_box.forward_process(torch.rand(3, 8), targets=torch.rand(3, 5))
        x2 = torch.rand(3, 8)
        training_box.forward_process(x2, targets=torch.rand(3, 5))
        hooked_output = training_box.criterion.captured_io_dict['student']['fc']['output']
        assert torch.equal(hooked_output, model(x2))
        training_box.clean_modules()

    def test_clean_modules_removes_hooks(self):
        model, training_box = self.build_box({'output': ['fc']})
        training_box.clean_modules()
        assert len(training_box.target_model_pairs) == 0
        assert len(training_box.model_io_dict) == 0
        model(torch.rand(3, 8))
        assert len(training_box.model_io_dict) == 0

    def test_multi_stage_box_updates_hooks_at_next_stage(self):
        model = ToyNet()
        model.eval()
        stage1_config = build_train_config(model_config={'forward_hook': {'output': ['layer1']}, 'forward_proc': 'forward_batch_only'})
        stage2_config = build_train_config(model_config={'forward_hook': {'input': ['fc'], 'output': ['fc']}, 'forward_proc': 'forward_batch_only'})
        train_config = {'stage1': stage1_config, 'stage2': stage2_config}
        training_box = get_training_box(
            model, dict(), train_config, torch.device('cpu'), None, False, 1.0
        )
        assert {module_path for module_path, _ in training_box.target_model_pairs} == {'layer1'}
        training_box.advance_to_next_stage()
        assert {module_path for module_path, _ in training_box.target_model_pairs} == {'fc'}
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_z2 = model.layer2(model.layer1(x))
        expected_y = model.fc(expected_z2)
        assert 'layer1' not in student_io_dict
        assert torch.equal(student_io_dict['fc']['input'], expected_z2)
        assert torch.equal(student_io_dict['fc']['output'], expected_y)
        training_box.clean_modules()


class DistillationBoxForwardHookUnitTest(TestCase):
    @staticmethod
    def build_box(teacher_forward_hook_config, student_forward_hook_config):
        teacher_model = ToyNet()
        student_model = ToyNet()
        teacher_model.eval()
        student_model.eval()
        train_config = build_train_config(
            teacher_model_config={'forward_hook': teacher_forward_hook_config, 'forward_proc': 'forward_batch_only',
                               'requires_grad': False},
            student_model_config={'forward_hook': student_forward_hook_config, 'forward_proc': 'forward_batch_only'}
        )
        distillation_box = get_distillation_box(
            teacher_model, student_model, dict(), train_config, torch.device('cpu'), None, False, 1.0
        )
        return teacher_model, student_model, distillation_box

    def test_set_hooks_registers_configured_module_paths(self):
        _, _, distillation_box = self.build_box({'output': ['layer2']}, {'input': ['fc'], 'output': ['layer1']})
        assert {module_path for module_path, _ in distillation_box.target_teacher_pairs} == {'layer2'}
        assert {module_path for module_path, _ in distillation_box.target_student_pairs} == {'fc', 'layer1'}
        assert set(distillation_box.teacher_io_dict.keys()) == {'layer2'}
        assert set(distillation_box.student_io_dict.keys()) == {'fc', 'layer1'}
        distillation_box.clean_modules()

    def test_extracts_teacher_and_student_outputs(self):
        teacher_model, student_model, distillation_box = \
            self.build_box({'output': ['layer2', 'fc']}, {'output': ['layer2', 'fc']})
        x = torch.rand(3, 8)
        distillation_box.forward_process(x, targets=torch.rand(3, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        expected_teacher_z2 = teacher_model.layer2(teacher_model.layer1(x))
        expected_teacher_y = teacher_model.fc(expected_teacher_z2)
        expected_student_z2 = student_model.layer2(student_model.layer1(x))
        expected_student_y = student_model.fc(expected_student_z2)
        assert torch.equal(io_dict['teacher']['layer2']['output'], expected_teacher_z2)
        assert torch.equal(io_dict['teacher']['fc']['output'], expected_teacher_y)
        assert torch.equal(io_dict['student']['layer2']['output'], expected_student_z2)
        assert torch.equal(io_dict['student']['fc']['output'], expected_student_y)
        # Teacher and student are separately initialized models, so their outputs should differ
        assert not torch.equal(expected_teacher_y, expected_student_y)
        distillation_box.clean_modules()

    def test_extracts_teacher_and_student_inputs(self):
        teacher_model, student_model, distillation_box = self.build_box({'input': ['fc']}, {'input': ['layer2']})
        x = torch.rand(3, 8)
        distillation_box.forward_process(x, targets=torch.rand(3, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        expected_teacher_fc_input = teacher_model.layer2(teacher_model.layer1(x))
        expected_student_layer2_input = student_model.layer1(x)
        assert torch.equal(io_dict['teacher']['fc']['input'], expected_teacher_fc_input)
        assert torch.equal(io_dict['student']['layer2']['input'], expected_student_layer2_input)
        assert 'output' not in io_dict['teacher']['fc']
        assert 'output' not in io_dict['student']['layer2']
        distillation_box.clean_modules()

    def test_io_dict_contains_model_outputs_at_self_module_path(self):
        teacher_model, student_model, distillation_box = self.build_box({'output': ['fc']}, {'output': ['fc']})
        x = torch.rand(3, 8)
        distillation_box.forward_process(x, targets=torch.rand(3, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        assert torch.equal(io_dict['teacher'][SELF_MODULE_PATH]['output'], teacher_model(x))
        assert torch.equal(io_dict['student'][SELF_MODULE_PATH]['output'], student_model(x))
        distillation_box.clean_modules()

    def test_hooks_are_independent_between_teacher_and_student(self):
        teacher_model, student_model, distillation_box = self.build_box({'output': ['layer1']}, {'output': ['fc']})
        x = torch.rand(3, 8)
        distillation_box.forward_process(x, targets=torch.rand(3, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        assert 'layer1' in io_dict['teacher'] and 'fc' not in io_dict['teacher']
        assert 'fc' in io_dict['student'] and 'layer1' not in io_dict['student']
        distillation_box.clean_modules()

    def test_clean_modules_removes_teacher_and_student_hooks(self):
        teacher_model, student_model, distillation_box = self.build_box({'output': ['fc']}, {'output': ['fc']})
        distillation_box.clean_modules()
        assert len(distillation_box.target_teacher_pairs) == 0
        assert len(distillation_box.target_student_pairs) == 0
        assert len(distillation_box.teacher_io_dict) == 0
        assert len(distillation_box.student_io_dict) == 0
        teacher_model(torch.rand(3, 8))
        student_model(torch.rand(3, 8))
        assert len(distillation_box.teacher_io_dict) == 0
        assert len(distillation_box.student_io_dict) == 0


class SharedModuleNet(nn.Module):
    """Model that calls the same submodule twice and has a module taking two inputs."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.merger = MergerModule()
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        z1 = self.shared(x)
        z2 = self.shared(z1)
        return self.fc(self.merger(z1, z2))


class MergerModule(nn.Module):
    """Module whose forward takes two positional arguments."""
    def forward(self, x1, x2):
        return x1 + x2


@register_auxiliary_model_wrapper
class ToyAuxiliaryModelWrapper(AuxiliaryModelWrapper):
    """Auxiliary model wrapper that post-processes hooked activations in :meth:`secondary_forward`."""
    def __init__(self, student_model=None, teacher_model=None, **kwargs):
        super().__init__()
        self.model = student_model if student_model is not None else teacher_model
        self.translator = nn.Linear(4, 4)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def secondary_forward(self, io_dict):
        # Emulates FT-style auxiliary modules that are fed hooked activations after the main forward
        self.translator(io_dict['model.layer2']['output'])


class ForwardHookAlternativePathUnitTest(TestCase):
    @staticmethod
    def build_training_box(model, model_config):
        model_config = dict(model_config)
        model_config.setdefault('forward_proc', 'forward_batch_only')
        return get_training_box(
            model, dict(), build_train_config(model_config=model_config),
            torch.device('cpu'), None, False, 1.0
        )

    def test_extracts_output_of_adaptation_module_added_by_sequential(self):
        model = ToyNet()
        model.eval()
        model_config = {
            'sequential': ['layer1', 'layer2', '+adapter'],
            'adaptations': {
                'adapter': {'key': 'Linear', 'kwargs': {'in_features': 4, 'out_features': 3}}
            },
            'forward_hook': {'output': ['layer1', '+adapter']}
        }
        training_box = self.build_training_box(model, model_config)
        # The redesigned model is a new sequential container ending with the adaptation module
        adapter = training_box.model.adapter
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 3))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_z1 = model.layer1(x)
        expected_adapter_output = adapter(model.layer2(expected_z1))
        assert torch.equal(student_io_dict['layer1']['output'], expected_z1)
        assert torch.equal(student_io_dict['+adapter']['output'], expected_adapter_output)
        assert torch.equal(
            student_io_dict[SELF_MODULE_PATH]['output'], expected_adapter_output
        )
        training_box.clean_modules()

    def test_extracts_output_of_nested_module_path(self):
        model = ToyNet()
        model.eval()
        training_box = self.build_training_box(model, {'forward_hook': {'output': ['layer1.0', 'layer2.1']}})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_linear_output = model.layer1[0](x)
        expected_relu_output = model.layer2(model.layer1(x))
        assert torch.equal(student_io_dict['layer1.0']['output'], expected_linear_output)
        assert torch.equal(student_io_dict['layer2.1']['output'], expected_relu_output)
        training_box.clean_modules()

    def test_extracts_last_call_of_module_called_multiple_times(self):
        model = SharedModuleNet()
        model.eval()
        training_box = self.build_training_box(model, {'forward_hook': {'input': ['shared'], 'output': ['shared']}})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        shared_io_dict = training_box.criterion.captured_io_dict['student']['shared']
        expected_z1 = model.shared(x)
        expected_z2 = model.shared(expected_z1)
        # Only the last of the two calls within a single forward pass is kept
        assert torch.equal(shared_io_dict['input'], expected_z1)
        assert torch.equal(shared_io_dict['output'], expected_z2)
        training_box.clean_modules()

    def test_extracts_input_of_module_taking_multiple_args_as_tuple(self):
        model = SharedModuleNet()
        model.eval()
        training_box = self.build_training_box(model, {'forward_hook': {'input': ['merger']}})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        merger_input = training_box.criterion.captured_io_dict['student']['merger']['input']
        expected_z1 = model.shared(x)
        expected_z2 = model.shared(expected_z1)
        assert isinstance(merger_input, tuple)
        assert len(merger_input) == 2
        assert torch.equal(merger_input[0], expected_z1)
        assert torch.equal(merger_input[1], expected_z2)
        training_box.clean_modules()

    def test_extracts_output_of_auxiliary_model_wrapper_module(self):
        model = ToyNet()
        model.eval()
        model_config = {
            'auxiliary_model_wrapper': {'key': 'ToyAuxiliaryModelWrapper'},
            # Module paths are relative to the auxiliary model wrapper
            'forward_hook': {'output': ['model.layer2', 'translator']}
        }
        training_box = self.build_training_box(model, model_config)
        assert isinstance(training_box.model, AuxiliaryModelWrapper)
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_z2 = model.layer2(model.layer1(x))
        expected_translator_output = training_box.model.translator(expected_z2)
        assert torch.equal(student_io_dict['model.layer2']['output'], expected_z2)
        # `translator` is only called in secondary_forward, so this checks the post-forward merge as well
        assert torch.equal(student_io_dict['translator']['output'], expected_translator_output)
        training_box.clean_modules()

    def test_teacher_extraction_is_detached_from_autograd_when_not_updatable(self):
        teacher_model = ToyNet()
        student_model = ToyNet()
        teacher_model.eval()
        student_model.eval()
        train_config = build_train_config(
            teacher_model_config={
                'forward_hook': {'output': ['layer2']},
                'forward_proc': 'forward_batch_only',
                'requires_grad': False
            },
            student_model_config={
                'forward_hook': {'output': ['layer2']},
                'forward_proc': 'forward_batch_only'
            }
        )
        distillation_box = get_distillation_box(
            teacher_model, student_model, dict(), train_config, torch.device('cpu'), None, False, 1.0
        )
        assert not distillation_box.teacher_updatable
        distillation_box.forward_process(torch.rand(3, 8), targets=torch.rand(3, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        assert not io_dict['teacher']['layer2']['output'].requires_grad
        assert not io_dict['teacher'][SELF_MODULE_PATH]['output'].requires_grad
        assert io_dict['student']['layer2']['output'].requires_grad
        assert io_dict['student'][SELF_MODULE_PATH]['output'].requires_grad
        distillation_box.clean_modules()


class ToyAutoregressiveNet(nn.Module):
    """Model that calls the same block once per generated step, as in autoregressive generation."""
    def __init__(self, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.embed = nn.Linear(8, 6)
        self.block = nn.Linear(6, 6)
        self.head = nn.Linear(6, 5)

    def forward(self, x):
        hidden = self.embed(x)
        step_outputs = list()
        for _ in range(self.num_steps):
            hidden = self.block(hidden)
            step_outputs.append(self.head(hidden))
        return torch.stack(step_outputs, dim=1)

    def step_hidden_states(self, x):
        """Recomputes the per-step outputs of `block` without relying on any hook."""
        hidden = self.embed(x)
        hidden_states = list()
        for _ in range(self.num_steps):
            hidden = self.block(hidden)
            hidden_states.append(hidden)
        return hidden_states


class AccumulatingForwardHookUnitTest(TestCase):
    @staticmethod
    def build_training_box(model, forward_hook_config):
        train_config = build_train_config(
            model_config={'forward_hook': forward_hook_config, 'forward_proc': 'forward_batch_only'}
        )
        return get_training_box(model, dict(), train_config, torch.device('cpu'), None, False, 1.0)

    def test_accumulates_per_step_outputs_as_list(self):
        model = ToyAutoregressiveNet()
        model.eval()
        training_box = self.build_training_box(model, {'output': ['block'], 'accumulates': ['block']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 4, 5))
        hooked_outputs = training_box.criterion.captured_io_dict['student']['block']['output']
        expected_hidden_states = model.step_hidden_states(x)
        assert isinstance(hooked_outputs, list)
        assert len(hooked_outputs) == model.num_steps
        for expected, actual in zip(expected_hidden_states, hooked_outputs):
            assert torch.equal(expected, actual)
        training_box.clean_modules()

    def test_stacks_accumulated_per_step_outputs(self):
        model = ToyAutoregressiveNet()
        model.eval()
        training_box = self.build_training_box(
            model, {'output': ['block'], 'accumulates': ['block'], 'stacks_accumulated': ['block']}
        )
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 4, 5))
        hooked_outputs = training_box.criterion.captured_io_dict['student']['block']['output']
        expected_hidden_states = model.step_hidden_states(x)
        assert isinstance(hooked_outputs, torch.Tensor)
        assert hooked_outputs.shape == (model.num_steps, 3, 6)
        for i, expected in enumerate(expected_hidden_states):
            assert torch.equal(expected, hooked_outputs[i])
        training_box.clean_modules()

    def test_accumulates_true_applies_to_all_target_modules(self):
        model = ToyAutoregressiveNet()
        model.eval()
        training_box = self.build_training_box(
            model, {'input': ['head'], 'output': ['block', 'head'], 'accumulates': True}
        )
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 4, 5))
        student_io_dict = training_box.criterion.captured_io_dict['student']
        expected_hidden_states = model.step_hidden_states(x)
        assert len(student_io_dict['block']['output']) == model.num_steps
        assert len(student_io_dict['head']['output']) == model.num_steps
        # Input of `head` at each step is the output of `block` at the same step
        for expected, actual in zip(expected_hidden_states, student_io_dict['head']['input']):
            assert torch.equal(expected, actual)
        training_box.clean_modules()

    def test_without_accumulates_only_last_step_is_kept(self):
        model = ToyAutoregressiveNet()
        model.eval()
        training_box = self.build_training_box(model, {'output': ['block']})
        x = torch.rand(3, 8)
        training_box.forward_process(x, targets=torch.rand(3, 4, 5))
        hooked_output = training_box.criterion.captured_io_dict['student']['block']['output']
        assert isinstance(hooked_output, torch.Tensor)
        assert torch.equal(hooked_output, model.step_hidden_states(x)[-1])
        training_box.clean_modules()

    def test_accumulated_outputs_do_not_leak_across_forward_processes(self):
        model = ToyAutoregressiveNet()
        model.eval()
        training_box = self.build_training_box(model, {'output': ['block'], 'accumulates': ['block']})
        training_box.forward_process(torch.rand(3, 8), targets=torch.rand(3, 4, 5))
        x2 = torch.rand(3, 8)
        training_box.forward_process(x2, targets=torch.rand(3, 4, 5))
        hooked_outputs = training_box.criterion.captured_io_dict['student']['block']['output']
        assert len(hooked_outputs) == model.num_steps
        for expected, actual in zip(model.step_hidden_states(x2), hooked_outputs):
            assert torch.equal(expected, actual)
        training_box.clean_modules()

    def test_accumulates_with_unhooked_module_path_raises(self):
        model = ToyAutoregressiveNet()
        model.eval()
        with self.assertRaises(ValueError):
            self.build_training_box(model, {'output': ['block'], 'accumulates': ['embed']})

    def test_stacks_accumulated_without_accumulates_raises(self):
        model = ToyAutoregressiveNet()
        model.eval()
        with self.assertRaises(ValueError):
            self.build_training_box(model, {'output': ['block'], 'stacks_accumulated': ['block']})

    def test_distillation_box_accumulates_teacher_and_student_per_step_outputs(self):
        teacher_model = ToyAutoregressiveNet()
        student_model = ToyAutoregressiveNet()
        teacher_model.eval()
        student_model.eval()
        train_config = build_train_config(
            teacher_model_config={
                'forward_hook': {'output': ['block'], 'accumulates': True, 'stacks_accumulated': True},
                'forward_proc': 'forward_batch_only',
                'requires_grad': False
            },
            student_model_config={
                'forward_hook': {'output': ['block'], 'accumulates': ['block']},
                'forward_proc': 'forward_batch_only'
            }
        )
        distillation_box = get_distillation_box(
            teacher_model, student_model, dict(), train_config, torch.device('cpu'), None, False, 1.0
        )
        x = torch.rand(3, 8)
        distillation_box.forward_process(x, targets=torch.rand(3, 4, 5))
        io_dict = distillation_box.criterion.captured_io_dict
        teacher_outputs = io_dict['teacher']['block']['output']
        student_outputs = io_dict['student']['block']['output']
        # The teacher stacks its accumulated steps while the student keeps them as a list
        assert isinstance(teacher_outputs, torch.Tensor)
        assert teacher_outputs.shape == (teacher_model.num_steps, 3, 6)
        assert isinstance(student_outputs, list)
        assert len(student_outputs) == student_model.num_steps
        for i, expected in enumerate(teacher_model.step_hidden_states(x)):
            assert torch.equal(expected, teacher_outputs[i])
        for expected, actual in zip(student_model.step_hidden_states(x), student_outputs):
            assert torch.equal(expected, actual)
        assert not teacher_outputs.requires_grad
        assert student_outputs[0].requires_grad
        distillation_box.clean_modules()


class ToyCausalLMOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class ToyCausalLM(nn.Module):
    """Causal LM with an explicit ``forward`` signature that records the kwargs of each call."""
    def __init__(self, vocab_size=10, generated_ids=None, eos_token_id=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.generated_ids = generated_ids
        self.call_kwargs_list = list()

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        self.call_kwargs_list.append(
            {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids, 'labels': labels}
        )
        logits = self.embed(input_ids)
        loss = None if labels is None else logits.sum() * 0.0 + labels.float().mean()
        return ToyCausalLMOutput(logits, loss)

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, **kwargs):
        assert not self.training
        return self.generated_ids


class ToyCausalLMWithKwargs(ToyCausalLM):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.call_kwargs_list.append({'input_ids': input_ids, 'attention_mask': attention_mask, **kwargs})
        return ToyCausalLMOutput(self.embed(input_ids))


class ToyCausalLMWithoutPositionIds(ToyCausalLM):
    def forward(self, input_ids, attention_mask=None):
        self.call_kwargs_list.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        return ToyCausalLMOutput(self.embed(input_ids))


class AutoregressiveForwardProcUnitTest(TestCase):
    def setUp(self):
        self.sample_batch = {
            'input_ids': torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            'labels': torch.tensor([[-100, 4, 5, -100], [-100, 7, -100, -100]]),
            'prompt_input_ids': torch.tensor([[0, 3], [6, 7]]),
            'prompt_attention_mask': torch.tensor([[0, 1], [1, 1]]),
        }

    def test_forward_batch_as_kwargs_drops_unaccepted_entries(self):
        model = ToyCausalLM()
        get_forward_proc_func('forward_batch_as_kwargs')(model, self.sample_batch)
        assert set(model.call_kwargs_list[0].keys()) == {'input_ids', 'attention_mask', 'position_ids', 'labels'}
        assert model.call_kwargs_list[0]['position_ids'] is None

    def test_forward_batch_as_kwargs_keeps_all_entries_for_var_kwargs(self):
        model = ToyCausalLMWithKwargs()
        get_forward_proc_func('forward_batch_as_kwargs')(model, self.sample_batch)
        assert set(model.call_kwargs_list[0].keys()) == set(self.sample_batch.keys())

    def test_forward_batch_as_kwargs_reads_signature_of_wrapped_model(self):
        model = ToyCausalLM()
        get_forward_proc_func('forward_batch_as_kwargs')(DataParallel(model), self.sample_batch)
        assert 'prompt_input_ids' not in model.call_kwargs_list[0]

    def test_generate_on_policy_sequences_masks_completion_after_first_eos(self):
        eos_token_id = 2
        # Left-padded prompts of width 2, followed by 4 generated elements padded with EOS after the first EOS
        generated_ids = torch.tensor([[0, 3, 8, 2, 2, 2], [6, 7, 8, 9, 9, 9]])
        model = ToyCausalLM(generated_ids=generated_ids, eos_token_id=eos_token_id)
        model.train()
        box = SimpleNamespace(student_model=model, accelerator=None)
        sample_batch = dict(self.sample_batch)
        get_pre_forward_proc_func('generate_on_policy_sequences')(box, sample_batch=sample_batch, max_new_tokens=4)
        assert model.training
        assert sample_batch['prompt_length'] == 2
        assert torch.equal(sample_batch['on_policy_input_ids'], generated_ids)
        expected_completion_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        assert torch.equal(sample_batch['completion_mask'], expected_completion_mask)
        assert torch.equal(
            sample_batch['on_policy_attention_mask'], torch.tensor([[0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        )

    def test_generate_on_policy_sequences_without_prompts_raises(self):
        box = SimpleNamespace(student_model=ToyCausalLM(), accelerator=None)
        with self.assertRaises(KeyError):
            get_pre_forward_proc_func('generate_on_policy_sequences')(box, sample_batch={'input_ids': None})

    def test_forward_on_policy_sequences_derives_position_ids_from_attention_mask(self):
        model = ToyCausalLM()
        sample_batch = {
            'on_policy_input_ids': torch.tensor([[0, 3, 8, 2], [6, 7, 8, 9]]),
            'on_policy_attention_mask': torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        }
        get_forward_proc_func('forward_on_policy_sequences')(model, sample_batch)
        assert torch.equal(model.call_kwargs_list[0]['position_ids'], torch.tensor([[0, 0, 1, 2], [0, 1, 2, 3]]))

    def test_forward_on_policy_sequences_omits_position_ids_if_not_accepted(self):
        model = ToyCausalLMWithoutPositionIds()
        sample_batch = {
            'on_policy_input_ids': torch.tensor([[0, 3, 8, 2]]),
            'on_policy_attention_mask': torch.tensor([[0, 1, 1, 1]]),
        }
        get_forward_proc_func('forward_on_policy_sequences')(model, sample_batch)
        assert 'position_ids' not in model.call_kwargs_list[0]

    def test_forward_on_policy_sequences_with_task_loss_scores_reference_sequences(self):
        model = ToyCausalLM()
        sample_batch = dict(self.sample_batch)
        sample_batch['on_policy_input_ids'] = torch.tensor([[0, 3, 8, 2], [6, 7, 8, 9]])
        sample_batch['on_policy_attention_mask'] = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        outputs = get_forward_proc_func('forward_on_policy_sequences_with_task_loss')(model, sample_batch)
        on_policy_kwargs, reference_kwargs = model.call_kwargs_list
        assert torch.equal(outputs.logits, model.embed(sample_batch['on_policy_input_ids']))
        assert torch.equal(on_policy_kwargs['input_ids'], sample_batch['on_policy_input_ids'])
        assert torch.equal(reference_kwargs['input_ids'], sample_batch['input_ids'])
        assert torch.equal(reference_kwargs['labels'], sample_batch['labels'])
        assert torch.equal(outputs.loss, sample_batch['labels'].float().mean())

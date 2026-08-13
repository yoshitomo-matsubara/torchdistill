from unittest import TestCase

import torch
from torch import nn
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
        return sum(loss.mean() for loss in model_loss_dict.values())


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

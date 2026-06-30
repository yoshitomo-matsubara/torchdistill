import copy
import functools
import json
import math
import os
from unittest import TestCase

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
import yaml
from PIL import Image

import torchdistill.losses.high_level  # noqa: registers high-level losses
import torchdistill.losses.mid_level  # noqa: registers mid-level losses
import torchdistill.models.wrapper  # noqa: registers auxiliary model wrappers
from torchdistill.core.forward_hook import ForwardHookManager
from torchdistill.core.util import extract_io_dict, update_io_dict
from torchdistill.losses.registry import get_high_level_loss
from torchdistill.models.wrapper import AuxiliaryModelWrapper

DEVICE = torch.device('cpu')
DEVICE_IDS = [0]
DISTRIBUTED = False
IMAGE_PATHS = [
    'tests/fixtures/imgs/random_400x600.jpg',
    'tests/fixtures/imgs/random_512x512.jpg',
    'tests/fixtures/imgs/random_600x400.jpg',
    'tests/fixtures/imgs/random_600x600.jpg',
]
CONFIG_ROOT = 'configs/sample/ilsvrc2012'
OUTPUT_PATH = 'tests/fixtures/losses/pre-computed_loss.json'


def load_yaml_raw(yaml_path):
    """Load a YAML file without executing custom constructors."""
    class RawLoader(yaml.SafeLoader):
        pass

    def handle_join(loader, node):
        seq = loader.construct_sequence(node, deep=True)
        return ''.join(str(s) for s in seq)

    def handle_unknown(loader, tag_suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    RawLoader.add_constructor('!join', handle_join)
    RawLoader.add_multi_constructor('!', handle_unknown)

    with open(yaml_path) as f:
        return yaml.load(f, Loader=RawLoader)


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@functools.lru_cache(maxsize=1)
def create_student():
    return tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1).eval()


@functools.lru_cache(maxsize=1)
def create_teacher():
    return tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1).eval()


def create_aux_wrapper(wrapper_key, wrapper_kwargs, teacher_model=None, student_model=None):
    from torchdistill.models.registry import get_auxiliary_model_wrapper
    kwargs = copy.deepcopy(wrapper_kwargs)
    kwargs['device'] = DEVICE
    kwargs['device_ids'] = DEVICE_IDS
    kwargs['distributed'] = DISTRIBUTED
    if teacher_model is not None:
        kwargs['teacher_model'] = teacher_model
    if student_model is not None:
        kwargs['student_model'] = student_model
    return get_auxiliary_model_wrapper(wrapper_key, **kwargs).eval()


def setup_hooks(model, hook_config, hook_manager):
    for path in hook_config.get('output', []):
        hook_manager.add_hook(model, path, requires_input=False, requires_output=True)
    for path in hook_config.get('input', []):
        hook_manager.add_hook(model, path, requires_input=True, requires_output=False)


def run_forward(model, x, hook_manager):
    """Run forward and secondary_forward (for AuxiliaryModelWrapper), return extracted io_dict."""
    with torch.no_grad():
        output = model(x)
    extracted = extract_io_dict(hook_manager.io_dict, DEVICE)
    extracted['.']['output'] = output
    if isinstance(model, AuxiliaryModelWrapper):
        with torch.no_grad():
            model.secondary_forward(extracted)
        update_io_dict(extracted, extract_io_dict(hook_manager.io_dict, DEVICE))
    return output, extracted


def run_config(yaml_path, student, teacher, image_tensor, targets, train_key=None):
    """Parse YAML, build hooks and criterion from config, return scalar loss value."""
    config = load_yaml_raw(yaml_path)
    train_config = config['train'][train_key] if train_key else config['train']

    teacher_config = train_config['teacher']
    student_config = train_config['student']
    criterion_config = train_config['criterion']

    teacher_hook_config = teacher_config.get('forward_hook') or {'input': [], 'output': []}
    student_hook_config = student_config.get('forward_hook') or {'input': [], 'output': []}

    teacher_aux = teacher_config.get('auxiliary_model_wrapper') or {}
    student_aux = student_config.get('auxiliary_model_wrapper') or {}
    teacher_wrapper_key = teacher_aux.get('key') if teacher_aux else None
    student_wrapper_key = student_aux.get('key') if student_aux else None
    teacher_wrapper_kwargs = copy.deepcopy(teacher_aux.get('kwargs') or {}) if teacher_aux else {}
    student_wrapper_kwargs = copy.deepcopy(student_aux.get('kwargs') or {}) if student_aux else {}

    teacher_model = create_aux_wrapper(teacher_wrapper_key, teacher_wrapper_kwargs, teacher_model=copy.deepcopy(teacher)) \
        if teacher_wrapper_key else copy.deepcopy(teacher)
    student_model = create_aux_wrapper(student_wrapper_key, student_wrapper_kwargs, student_model=copy.deepcopy(student)) \
        if student_wrapper_key else copy.deepcopy(student)

    teacher_hook_mgr = ForwardHookManager(DEVICE)
    student_hook_mgr = ForwardHookManager(DEVICE)
    setup_hooks(teacher_model, teacher_hook_config, teacher_hook_mgr)
    setup_hooks(student_model, student_hook_config, student_hook_mgr)

    _, teacher_io_dict = run_forward(teacher_model, image_tensor, teacher_hook_mgr)
    _, student_io_dict = run_forward(student_model, image_tensor, student_hook_mgr)

    criterion = get_high_level_loss(criterion_config)
    io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
    with torch.no_grad():
        loss = criterion(io_dict, {}, targets)
    return loss.item()


class MidLevelLossTest(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.transform = get_train_transform()
        tensors = [cls.transform(Image.open(p).convert('RGB')) for p in IMAGE_PATHS]
        cls.image_tensor = torch.stack(tensors)          # (4, 3, 224, 224)
        cls.targets = torch.zeros(4, dtype=torch.long)
        # SSKDLoss requires targets of size batch_size // 4 (one per original image)
        cls.sskd_target = torch.zeros(1, dtype=torch.long)
        cls.student = create_student()
        cls.teacher = create_teacher()
        cls.results = {}
        try:
            with open(OUTPUT_PATH) as f:
                cls.expected_losses = json.load(f)
        except FileNotFoundError:
            cls.expected_losses = {}

    @classmethod
    def tearDownClass(cls):
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(cls.results, f, indent=2)

    def _record(self, key, value):
        self.results[key] = value

    def _assert_loss_close(self, key, computed):
        expected = self.expected_losses.get(key)
        if expected is None:
            return
        if math.isnan(expected):
            self.assertTrue(math.isnan(computed), f'{key}: expected NaN but got {computed}')
        else:
            self.assertTrue(
                math.isclose(computed, expected, rel_tol=1e-5, abs_tol=1e-6),
                f'{key}: computed {computed}, expected {expected}',
            )

    def test_at(self):
        key = 'at/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/at/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_cckd(self):
        key = 'cckd/resnet18_from_resnet50'
        yaml_path = f'{CONFIG_ROOT}/cckd/resnet18_from_resnet50.yaml'
        config = load_yaml_raw(yaml_path)
        train_config = config['train']
        teacher_config = train_config['teacher']
        student_config = train_config['student']
        criterion_config = copy.deepcopy(train_config['criterion'])

        # Adapt teacher Linear4CCKD for ResNet-34 (512 features instead of ResNet-50's 2048)
        teacher_wrapper_kwargs = copy.deepcopy(teacher_config['auxiliary_model_wrapper']['kwargs'])
        teacher_wrapper_kwargs['linear_kwargs']['in_features'] = 512

        # Fix criterion: YAML uses 'kernel_params'/'key' but code needs 'kernel_config'/'type'
        cckd_kwargs = criterion_config['kwargs']['sub_terms']['cckd']['criterion']['kwargs']
        cckd_kwargs['kernel_config'] = cckd_kwargs.pop('kernel_params')
        cckd_kwargs['kernel_config']['type'] = cckd_kwargs['kernel_config'].pop('key')

        teacher_model = create_aux_wrapper(
            'Linear4CCKD', teacher_wrapper_kwargs, teacher_model=copy.deepcopy(self.teacher)
        )
        student_model = create_aux_wrapper(
            'Linear4CCKD',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_cse_l2(self):
        key = 'cse_l2/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/cse_l2/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_dab(self):
        key = 'dab/resnet18_from_resnet50'
        yaml_path = f'{CONFIG_ROOT}/dab/resnet18_from_resnet50.yaml'
        config = load_yaml_raw(yaml_path)
        # Use stage1 which contains the AltActTransferLoss (DAB-specific loss)
        stage1_config = config['train']['stage1']
        teacher_config = stage1_config['teacher']
        student_config = stage1_config['student']
        criterion_config = stage1_config['criterion']

        # Adapt connectors: out_channels should match ResNet-34 teacher (64/128/256/512)
        # instead of ResNet-50 (256/512/1024/2048)
        teacher34_channels = {'connector1': 64, 'connector2': 128, 'connector3': 256, 'connector4': 512}
        student_wrapper_kwargs = copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs'])
        for cname, out_ch in teacher34_channels.items():
            conn = student_wrapper_kwargs['connectors'][cname]
            conn['conv2d_kwargs']['out_channels'] = out_ch
            if conn.get('bn2d_kwargs'):
                conn['bn2d_kwargs']['num_features'] = out_ch

        teacher_model = copy.deepcopy(self.teacher)
        student_model = create_aux_wrapper(
            'Connector4DAB', student_wrapper_kwargs, student_model=copy.deepcopy(self.student)
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_dist(self):
        key = 'dist/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/dist/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_dist_plus(self):
        key = 'dist_plus/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/dist_plus/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_fitnet_stage2(self):
        key = 'fitnet/resnet18_from_resnet152'
        loss = run_config(
            f'{CONFIG_ROOT}/fitnet/resnet18_from_resnet152.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
            train_key='stage2',
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_fsp(self):
        key = 'fsp/resnet18_from_resnet34'
        # Use stage1 which contains FSPLoss
        loss = run_config(
            f'{CONFIG_ROOT}/fsp/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
            train_key='stage1',
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_ft_stage2(self):
        key = 'ft/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/ft/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        stage2_config = config['train']['stage2']
        teacher_config = stage2_config['teacher']
        student_config = stage2_config['student']
        criterion_config = stage2_config['criterion']

        teacher_model = create_aux_wrapper(
            'Teacher4FactorTransfer',
            copy.deepcopy(teacher_config['auxiliary_model_wrapper']['kwargs']),
            teacher_model=copy.deepcopy(self.teacher),
        )
        student_model = create_aux_wrapper(
            'Student4FactorTransfer',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_ickd(self):
        key = 'ickd/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/ickd/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        train_config = config['train']
        teacher_config = train_config['teacher']
        student_config = train_config['student']
        criterion_config = train_config['criterion']

        teacher_model = copy.deepcopy(self.teacher)
        student_model = create_aux_wrapper(
            'Student4ICKD',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_kd_alexnet(self):
        key = 'kd/alexnet_from_resnet152'
        loss = run_config(
            f'{CONFIG_ROOT}/kd/alexnet_from_resnet152.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_kd_resnet18(self):
        key = 'kd/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/kd/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_kd_w_ls(self):
        key = 'kd_w_ls/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/kd_w_ls/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_kr(self):
        key = 'kr/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/kr/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        train_config = config['train']
        teacher_config = train_config['teacher']
        student_config = train_config['student']
        criterion_config = train_config['criterion']

        teacher_model = copy.deepcopy(self.teacher)
        student_model = create_aux_wrapper(
            'Student4KnowledgeReview',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_pad_stage2(self):
        key = 'pad/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/pad/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        stage2_config = config['train']['stage2']
        teacher_config = stage2_config['teacher']
        student_config = stage2_config['student']
        criterion_config = stage2_config['criterion']

        teacher_model = copy.deepcopy(self.teacher)
        student_model = create_aux_wrapper(
            'VarianceBranch4PAD',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_pkt(self):
        key = 'pkt/resnet18_from_resnet152'
        loss = run_config(
            f'{CONFIG_ROOT}/pkt/resnet18_from_resnet152.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_rkd(self):
        key = 'rkd/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/rkd/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_spkd(self):
        key = 'spkd/resnet18_from_resnet34'
        loss = run_config(
            f'{CONFIG_ROOT}/spkd/resnet18_from_resnet34.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_srd(self):
        key = 'srd/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/srd/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        train_config = config['train']
        teacher_config = train_config['teacher']
        student_config = train_config['student']
        criterion_config = train_config['criterion']

        teacher_model = create_aux_wrapper(
            'SRDModelWrapper',
            copy.deepcopy(teacher_config['auxiliary_model_wrapper']['kwargs']),
            teacher_model=copy.deepcopy(self.teacher),
        )
        student_model = create_aux_wrapper(
            'SRDModelWrapper',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_sskd_stage2(self):
        key = 'sskd/resnet18_from_resnet34'
        yaml_path = f'{CONFIG_ROOT}/sskd/resnet18_from_resnet34.yaml'
        config = load_yaml_raw(yaml_path)
        stage2_config = config['train']['stage2']
        teacher_config = stage2_config['teacher']
        student_config = stage2_config['student']
        criterion_config = stage2_config['criterion']

        teacher_model = create_aux_wrapper(
            'SSWrapper4SSKD',
            copy.deepcopy(teacher_config['auxiliary_model_wrapper']['kwargs']),
            teacher_model=copy.deepcopy(self.teacher),
        )
        student_model = create_aux_wrapper(
            'SSWrapper4SSKD',
            copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs']),
            student_model=copy.deepcopy(self.student),
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        # SSKDLoss expects batch_size % 4 == 0, structured as [orig, aug, aug, aug, ...].
        # The 4-image batch maps to 1 original (index 0) + 3 augmented (indices 1-3).
        # targets must have batch_size // 4 = 1 entry.
        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.sskd_target)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

    def test_tfkd(self):
        key = 'tfkd/resnet18_from_resnet18'
        loss = run_config(
            f'{CONFIG_ROOT}/tfkd/resnet18_from_resnet18.yaml',
            self.student, self.teacher, self.image_tensor, self.targets,
        )
        self._record(key, loss)
        self._assert_loss_close(key, loss)

    def test_vid(self):
        key = 'vid/resnet18_from_resnet50'
        yaml_path = f'{CONFIG_ROOT}/vid/resnet18_from_resnet50.yaml'
        config = load_yaml_raw(yaml_path)
        train_config = config['train']
        teacher_config = train_config['teacher']
        student_config = train_config['student']
        criterion_config = train_config['criterion']

        student_wrapper_kwargs = copy.deepcopy(student_config['auxiliary_model_wrapper']['kwargs'])
        # Remap 'regressor_kwargs' -> 'kwargs' to match wrapper code expectation.
        # Adapt out_channels/middle_channels to match ResNet-34 teacher channels (64/128/256/512).
        teacher34_channels = {'regressor1': 64, 'regressor2': 128, 'regressor3': 256, 'regressor4': 512}
        for rname, out_ch in teacher34_channels.items():
            reg_config = student_wrapper_kwargs['regressors'][rname]
            rk = reg_config.pop('regressor_kwargs')
            rk['out_channels'] = out_ch
            rk['middle_channels'] = out_ch
            reg_config['kwargs'] = rk

        teacher_model = copy.deepcopy(self.teacher)
        student_model = create_aux_wrapper(
            'VariationalDistributor4VID', student_wrapper_kwargs, student_model=copy.deepcopy(self.student)
        )

        teacher_hook_mgr = ForwardHookManager(DEVICE)
        student_hook_mgr = ForwardHookManager(DEVICE)
        setup_hooks(teacher_model, teacher_config['forward_hook'], teacher_hook_mgr)
        setup_hooks(student_model, student_config['forward_hook'], student_hook_mgr)

        _, teacher_io_dict = run_forward(teacher_model, self.image_tensor, teacher_hook_mgr)
        _, student_io_dict = run_forward(student_model, self.image_tensor, student_hook_mgr)

        criterion = get_high_level_loss(criterion_config)
        io_dict = {'teacher': teacher_io_dict, 'student': student_io_dict}
        with torch.no_grad():
            loss = criterion(io_dict, {}, self.targets)
        self._record(key, loss.item())
        self._assert_loss_close(key, loss.item())

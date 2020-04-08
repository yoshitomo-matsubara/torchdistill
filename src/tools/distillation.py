import sys

from torch import nn

from myutils.pytorch.func_util import get_optimizer, get_scheduler
from myutils.pytorch.module_util import check_if_wrapped, get_module, unfreeze_module_params
from tools.loss import KDLoss, get_single_loss, get_custom_loss
from utils.dataset_util import build_data_loaders
from utils.model_util import redesign_model, wrap_model

try:
    from apex import amp
except ImportError:
    amp = None


def extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def set_distillation_box_info(info_dict, module_path, **kwargs):
    info_dict[module_path] = kwargs


def register_forward_hook_with_dict(module, module_path, info_dict):
    def forward_hook(self, input, output):
        info_dict[module_path]['output'] = output
    return module.register_forward_hook(forward_hook)


class DistillationBox(nn.Module):
    def setup(self, train_config):
        # Set up train and val data loaders
        train_data_loader_config = train_config.get('train_data_loader', dict())
        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader =\
            build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config], self.distributed)
        if train_data_loader is not None:
            self.train_data_loader = train_data_loader
        if val_data_loader is not None:
            self.val_data_loader = val_data_loader

        # Define teacher and student models used in this stage
        unwrapped_org_teacher_model =\
            self.org_teacher_model.module if check_if_wrapped(self.org_teacher_model) else self.org_teacher_model
        unwrapped_org_student_model = \
            self.org_student_model.module if check_if_wrapped(self.org_student_model) else self.org_student_model
        self.target_module_pairs.clear()
        self.target_module_handles.clear()
        teacher_config = train_config.get('teacher', None)
        self.teacher_model = self.org_teacher_model if teacher_config is None \
            else redesign_model(unwrapped_org_teacher_model, teacher_config, 'teacher')
        student_config = train_config.get('student', None)
        self.student_model = self.org_student_model if student_config is None \
            else redesign_model(unwrapped_org_student_model, student_config, 'student')

        # Define loss function used in this stage
        criterion_config = train_config['criterion']
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            for loss_name, loss_config in sub_terms_config.items():
                teacher_path, student_path = loss_config['ts_modules']
                self.target_module_pairs.append((teacher_path, student_path))
                teacher_module = extract_module(unwrapped_org_teacher_model, self.teacher_model, teacher_path)
                student_module = extract_module(unwrapped_org_student_model, self.student_model, student_path)
                set_distillation_box_info(self.teacher_info_dict, teacher_path, loss_name=loss_name,
                                          path_from_root=teacher_path, is_teacher=True)
                set_distillation_box_info(self.student_info_dict, student_path, loss_name=loss_name,
                                          path_from_root=student_path, is_teacher=False)
                teacher_handle = register_forward_hook_with_dict(teacher_module, teacher_path, self.teacher_info_dict)
                student_handle = register_forward_hook_with_dict(student_module, student_path, self.student_info_dict)
                self.target_module_handles.append((teacher_handle, student_handle))

        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        if org_criterion_config is not None and len(org_criterion_config) > 0:
            self.org_criterion = get_single_loss(org_criterion_config)

        self.criterion = get_custom_loss(criterion_config)
        self.use_teacher_output = self.org_criterion is not None and isinstance(self.org_criterion, KDLoss)

        # Wrap models if necessary
        self.teacher_model = wrap_model(self.teacher_model, teacher_config, self.device, self.device_ids)
        self.student_model = wrap_model(self.student_model, student_config, self.device, self.device_ids)

        # Set up optimizer and scheduler
        optim_config = train_config['optimizer']
        self.optimizer = get_optimizer(self.student_model, optim_config['type'], optim_config['params'])
        scheduler_config = train_config.get('scheduler', None)
        self.lr_scheduler = None if scheduler_config is None \
            else get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])

        # Set up apex if you require mixed-precision training
        self.apex = False
        apex_config = train_config.get('apex', None)
        if apex_config is not None and apex_config.get('requires', False):
            if sys.version_info < (3, 0):
                raise RuntimeError('Apex currently only supports Python 3. Aborting.')
            if amp is None:
                raise RuntimeError('Failed to import apex. Please install apex from https://www.github.com/nvidia/apex '
                                   'to enable mixed-precision training.')
            self.student_model, self.optimizer =\
                amp.initialize(self.student_model, self.optimizer, opt_level=apex_config['opt_level'])
            self.apex = True

    def __init__(self, teacher_model, student_model, dataset_dict, train_config, device, device_ids, distributed):
        super().__init__()
        self.org_teacher_model = teacher_model.cpu()
        self.org_student_model = student_model.cpu()
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.teacher_model = None
        self.student_model = None
        self.target_module_pairs, self.target_module_handles = list(), list()
        self.teacher_info_dict, self.student_info_dict = dict(), dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.org_criterion, self.criterion, self.use_teacher_output = None, None, None
        self.apex = None
        self.setup(train_config)

    def pre_process(self, epoch=None, **kwargs):
        if self.distributed:
            self.train_data_loader.sampler.set_epoch(epoch)

    def check_if_org_loss_required(self):
        return self.org_criterion is not None

    def update_params(self, loss):
        self.optimizer.zero_grad()
        if self.apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

    def forward(self, sample_batch, targets):
        teacher_outputs = self.teacher_model(sample_batch)
        student_outputs = self.student_model(sample_batch)
        org_loss_dict = dict()
        if self.check_if_org_loss_required():
            # Models with auxiliary classifier returns multiple outputs
            if isinstance(student_outputs, (list, tuple)):
                if self.use_teacher_output:
                    for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                        org_loss_dict[i] = self.org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
                else:
                    for i, sub_outputs in enumerate(student_outputs):
                        org_loss_dict[i] = self.org_criterion(sub_outputs, targets)
            else:
                org_loss = self.org_criterion(student_outputs, teacher_outputs, targets) if self.use_teacher_output\
                    else self.org_criterion(student_outputs, targets)
                org_loss_dict = {0: org_loss}

        output_dict = dict()
        for teacher_path, student_path in self.target_module_pairs:
            teacher_module_dict = self.teacher_info_dict[teacher_path]
            student_module_dict = self.student_info_dict[student_path]
            output_dict[teacher_module_dict['loss_name']] = (
                (teacher_module_dict['path_from_root'], teacher_module_dict.pop('output')),
                (student_module_dict['path_from_root'], student_module_dict.pop('output'))
            )

        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss

    def post_process(self, **kwargs):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def clean_modules(self):
        unfreeze_module_params(self.org_teacher_model)
        unfreeze_module_params(self.org_student_model)
        for teacher_handle, student_handle in self.target_module_handles:
            teacher_handle.remove()
            student_handle.remove()


class MultiStagesDistillationBox(DistillationBox):
    def __init__(self, teacher_model, student_model, data_loader_dict, train_config, device, device_ids, distributed):
        stage1_config = train_config['stage1']
        super().__init__(teacher_model, student_model, data_loader_dict, stage1_config, device, device_ids, distributed)
        self.train_config = train_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['end_epoch']
        print('Stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        self.clean_modules()
        self.stage_number += 1
        next_stage_config = self.train_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config)
        self.stage_end_epoch = next_stage_config['end_epoch']
        print('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, epoch, **kwargs):
        if epoch == self.stage_end_epoch:
            self.advance_to_next_stage()


def get_distillation_box(teacher_model, student_model, data_loader_dict, train_config, device, device_ids, distributed):
    if 'stage1' in train_config:
        return MultiStagesDistillationBox(teacher_model, student_model, data_loader_dict,
                                          train_config, device, device_ids, distributed)
    return DistillationBox(teacher_model, student_model, data_loader_dict, train_config,
                           device, device_ids, distributed)

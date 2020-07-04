import copy
import sys

import torch
from torch import nn

from kdkit.common.constant import def_logger
from kdkit.datasets.util import build_data_loaders
from kdkit.models.special import SpecialModule, build_special_module
from kdkit.models.util import redesign_model
from kdkit.tools.foward_proc import get_forward_proc_func
from kdkit.tools.loss import KDLoss, get_single_loss, get_custom_loss, get_func2extract_org_output
from kdkit.tools.util import set_hooks, wrap_model, change_device, tensor2numpy2tensor, extract_outputs, \
    extract_sub_model_output_dict
from myutils.common.file_util import make_parent_dirs
from myutils.pytorch.func_util import get_optimizer, get_scheduler
from myutils.pytorch.module_util import check_if_wrapped, freeze_module_params, unfreeze_module_params

logger = def_logger.getChild(__name__)
try:
    from apex import amp
except ImportError:
    amp = None


class DistillationBox(nn.Module):
    def setup_data_loaders(self, train_config):
        train_data_loader_config = train_config.get('train_data_loader', dict())
        train_data_loader_config['requires_supp'] = True
        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader =\
            build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config], self.distributed)
        if train_data_loader is not None:
            self.train_data_loader = train_data_loader
        if val_data_loader is not None:
            self.val_data_loader = val_data_loader

    def setup_teacher_student_models(self, teacher_config, student_config):
        unwrapped_org_teacher_model =\
            self.org_teacher_model.module if check_if_wrapped(self.org_teacher_model) else self.org_teacher_model
        unwrapped_org_student_model = \
            self.org_student_model.module if check_if_wrapped(self.org_student_model) else self.org_student_model
        self.target_teacher_pairs.clear()
        self.target_student_pairs.clear()
        teacher_ref_model = unwrapped_org_teacher_model
        student_ref_model = unwrapped_org_student_model
        if len(teacher_config) > 0 or (len(teacher_config) == 0 and self.teacher_model is None):
            model_type = 'original'
            special_teacher_model = build_special_module(teacher_config, teacher_model=unwrapped_org_teacher_model)
            if special_teacher_model is not None:
                teacher_ref_model = special_teacher_model
                model_type = type(teacher_ref_model).__name__
            self.teacher_model = redesign_model(teacher_ref_model, teacher_config, 'teacher', model_type)

        if len(student_config) > 0 or (len(student_config) == 0 and self.student_model is None):
            model_type = 'original'
            special_student_model = build_special_module(student_config, student_model=unwrapped_org_student_model)
            if special_student_model is not None:
                student_ref_model = special_student_model
                model_type = type(student_ref_model).__name__
            self.student_model = redesign_model(student_ref_model, student_config, 'student', model_type)

        self.teacher_any_frozen = \
            len(teacher_config.get('frozen_modules', list())) > 0 or not teacher_config.get('requires_grad', True)
        self.student_any_frozen = \
            len(student_config.get('frozen_modules', list())) > 0 or not teacher_config.get('requires_grad', True)
        self.target_teacher_pairs.extend(set_hooks(self.teacher_model, teacher_ref_model,
                                                   teacher_config, self.teacher_info_dict))
        self.target_student_pairs.extend(set_hooks(self.student_model, student_ref_model,
                                                   student_config, self.student_info_dict))
        self.teacher_forward_proc = get_forward_proc_func(teacher_config.get('forward_proc', None))
        self.student_forward_proc = get_forward_proc_func(student_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        self.org_criterion = None if org_criterion_config is None or len(org_criterion_config) == 0 \
            else get_single_loss(org_criterion_config)
        self.criterion = get_custom_loss(criterion_config)
        self.uses_teacher_output = self.org_criterion is not None and isinstance(self.org_criterion, KDLoss)
        self.extract_org_loss = get_func2extract_org_output(criterion_config.get('func2extract_org_loss', None))

    def setup(self, train_config):
        # Set up train and val data loaders
        self.setup_data_loaders(train_config)

        # Define teacher and student models used in this stage
        teacher_config = train_config.get('teacher', dict())
        student_config = train_config.get('student', dict())
        self.setup_teacher_student_models(teacher_config, student_config)

        # Define loss function used in this stage
        self.setup_loss(train_config)

        # Wrap models if necessary
        self.teacher_model =\
            wrap_model(self.teacher_model, teacher_config, self.device, self.device_ids, self.distributed,
                       self.teacher_any_frozen)
        self.student_model =\
            wrap_model(self.student_model, student_config, self.device, self.device_ids, self.distributed,
                       self.student_any_frozen)
        self.teacher_updatable = True
        if not teacher_config.get('requires_grad', True):
            logger.info('Freezing the whole teacher model')
            freeze_module_params(self.teacher_model)
            self.teacher_updatable = False

        if not student_config.get('requires_grad', True):
            logger.info('Freezing the whole student model')
            freeze_module_params(self.student_model)

        # Set up optimizer and scheduler
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        trainable_module_list = nn.ModuleList([self.student_model])
        if self.teacher_updatable:
            logger.info('Note that you are training some/all of the modules in the teacher model')
            trainable_module_list.append(self.teacher_model)

        if len(optim_config) > 0:
            optim_params_config = optim_config['params']
            optim_params_config['lr'] *= self.lr_factor
            self.optimizer = get_optimizer(trainable_module_list, optim_config['type'], optim_params_config)
            optimizer_reset = True

        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
        elif optimizer_reset:
            self.lr_scheduler = None

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

    def __init__(self, teacher_model, student_model, dataset_dict,
                 train_config, device, device_ids, distributed, lr_factor):
        super().__init__()
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
        self.teacher_model = None
        self.student_model = None
        self.teacher_forward_proc, self.student_forward_proc = None, None
        self.target_teacher_pairs, self.target_student_pairs = list(), list()
        self.teacher_info_dict, self.student_info_dict = dict(), dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.org_criterion, self.criterion, self.uses_teacher_output, self.extract_org_loss = None, None, None, None
        self.teacher_updatable, self.teacher_any_frozen, self.student_any_frozen = None, None, None
        self.apex = None
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_process(self, epoch=None, **kwargs):
        self.teacher_model.eval()
        self.student_model.train()
        if self.distributed:
            self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)

    def get_teacher_output(self, sample_batch, targets, supp_dict):
        cached_data = supp_dict.get('cached_data', None)
        cache_file_paths = supp_dict.get('cache_file_path', None)
        teacher_outputs = None
        cached_extracted_teacher_output_dict = None
        # Use cached data if available
        if cached_data is not None and isinstance(cached_data, dict):
            device = sample_batch.device
            teacher_outputs = cached_data['teacher_outputs']
            cached_extracted_teacher_output_dict = cached_data['extracted_outputs']
            if device.type != 'cpu':
                teacher_outputs = change_device(teacher_outputs, device)
                cached_extracted_teacher_output_dict = change_device(cached_extracted_teacher_output_dict, device)
            if not self.teacher_updatable:
                return teacher_outputs, cached_extracted_teacher_output_dict

        if teacher_outputs is None:
            if self.teacher_updatable:
                teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)
            else:
                with torch.no_grad():
                    teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)

        if cached_extracted_teacher_output_dict is not None:
            if isinstance(self.teacher_model, SpecialModule) or \
                    (check_if_wrapped(self.teacher_model) and isinstance(self.teacher_model.module, SpecialModule)):
                self.teacher_info_dict.update(cached_extracted_teacher_output_dict)
                if isinstance(self.teacher_model, SpecialModule):
                    self.teacher_model.post_forward(self.teacher_info_dict)
                else:
                    self.teacher_model.module.post_forward(self.teacher_info_dict)

            extracted_teacher_output_dict = extract_outputs(self.teacher_info_dict)
            return teacher_outputs, extracted_teacher_output_dict

        # Deep copy of teacher info dict if teacher special module contains trainable module(s)
        teacher_info_dict4cache = copy.deepcopy(self.teacher_info_dict) \
            if self.teacher_updatable and isinstance(cache_file_paths, (list, tuple)) is not None else None
        if isinstance(self.teacher_model, SpecialModule):
            self.teacher_model.post_forward(self.teacher_info_dict)
        elif check_if_wrapped(self.teacher_model) and isinstance(self.teacher_model.module, SpecialModule):
            self.teacher_model.module.post_forward(self.teacher_info_dict)

        extracted_teacher_output_dict = extract_outputs(self.teacher_info_dict)
        # Write cache files if output file paths (cache_file_paths) are given
        if isinstance(cache_file_paths, (list, tuple)):
            if teacher_info_dict4cache is None:
                teacher_info_dict4cache = extracted_teacher_output_dict

            cpu_device = torch.device('cpu')
            for i, (teacher_output, cache_file_path) in enumerate(zip(teacher_outputs.cpu().numpy(), cache_file_paths)):
                sub_dict = extract_sub_model_output_dict(teacher_info_dict4cache, i)
                sub_dict = tensor2numpy2tensor(sub_dict, cpu_device)
                cache_dict = {'teacher_outputs': torch.Tensor(teacher_output), 'extracted_outputs': sub_dict}
                make_parent_dirs(cache_file_path)
                torch.save(cache_dict, cache_file_path)
        return teacher_outputs, extracted_teacher_output_dict

    def forward(self, sample_batch, targets, supp_dict):
        teacher_outputs, extracted_teacher_output_dict =\
            self.get_teacher_output(sample_batch, targets, supp_dict=supp_dict)
        student_outputs = self.student_forward_proc(self.student_model, sample_batch, targets, supp_dict)
        if isinstance(self.student_model, SpecialModule):
            self.student_model.post_forward(self.student_info_dict)
        elif check_if_wrapped(self.student_model) and isinstance(self.student_model.module, SpecialModule):
            self.student_model.module.post_forward(self.student_info_dict)

        org_loss_dict = self.extract_org_loss(self.org_criterion, student_outputs, teacher_outputs, targets,
                                              uses_teacher_output=self.uses_teacher_output, supp_dict=supp_dict)
        output_dict = {'teacher': extracted_teacher_output_dict,
                       'student': extract_outputs(self.student_info_dict)}
        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss

    def update_params(self, loss):
        self.optimizer.zero_grad()
        if self.apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

    def post_process(self, **kwargs):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if isinstance(self.teacher_model, SpecialModule):
            self.teacher_model.post_process()
        elif check_if_wrapped(self.teacher_model) and isinstance(self.teacher_model.module, SpecialModule):
            self.teacher_model.module.post_process()
        if isinstance(self.student_model, SpecialModule):
            self.student_model.post_process()
        elif check_if_wrapped(self.student_model) and isinstance(self.student_model.module, SpecialModule):
            self.student_model.module.post_process()

    def clean_modules(self):
        unfreeze_module_params(self.org_teacher_model)
        unfreeze_module_params(self.org_student_model)
        self.teacher_info_dict.clear()
        self.student_info_dict.clear()
        for _, module_handle in self.target_teacher_pairs + self.target_student_pairs:
            module_handle.remove()


class MultiStagesDistillationBox(DistillationBox):
    def __init__(self, teacher_model, student_model, data_loader_dict,
                 train_config, device, device_ids, distributed, lr_factor):
        stage1_config = train_config['stage1']
        super().__init__(teacher_model, student_model, data_loader_dict,
                         stage1_config, device, device_ids, distributed, lr_factor)
        self.train_config = train_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['num_epochs']
        self.num_epochs = sum(train_config[key]['num_epochs'] for key in train_config.keys() if key.startswith('stage'))
        self.current_epoch = 0
        logger.info('Started stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        self.clean_modules()
        self.stage_number += 1
        next_stage_config = self.train_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config)
        self.stage_end_epoch += next_stage_config['num_epochs']
        logger.info('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, **kwargs):
        super().post_process()
        self.current_epoch += 1
        if self.current_epoch == self.stage_end_epoch and self.current_epoch < self.num_epochs:
            self.advance_to_next_stage()


def get_distillation_box(teacher_model, student_model, data_loader_dict,
                         train_config, device, device_ids, distributed, lr_factor):
    if 'stage1' in train_config:
        return MultiStagesDistillationBox(teacher_model, student_model, data_loader_dict,
                                          train_config, device, device_ids, distributed, lr_factor)
    return DistillationBox(teacher_model, student_model, data_loader_dict, train_config,
                           device, device_ids, distributed, lr_factor)

import copy

import torch
from torch import nn

from .post_epoch_proc import default_post_epoch_process_with_teacher
from .post_forward_proc import default_post_forward_process
from .pre_epoch_proc import default_pre_epoch_process_with_teacher
from .pre_forward_proc import default_pre_forward_process
from .registry import get_pre_epoch_proc_func, get_pre_forward_proc_func, get_forward_proc_func, \
    get_post_forward_proc_func, get_post_epoch_proc_func
from .util import set_hooks, wrap_model, change_device, tensor2numpy2tensor, extract_io_dict, update_io_dict, \
    extract_sub_model_output_dict
from ..common.constant import SELF_MODULE_PATH, def_logger
from ..common.file_util import make_parent_dirs
from ..common.module_util import check_if_wrapped, freeze_module_params, get_module, \
    unfreeze_module_params, get_updatable_param_names
from ..datasets.util import build_data_loaders
from ..losses.registry import get_high_level_loss, get_func2extract_model_output
from ..models.util import redesign_model
from ..models.wrapper import AuxiliaryModelWrapper, build_auxiliary_model_wrapper
from ..optim.registry import get_optimizer, get_scheduler

logger = def_logger.getChild(__name__)


class DistillationBox(nn.Module):
    def setup_data_loaders(self, train_config):
        train_data_loader_config = train_config.get('train_data_loader', dict())
        if 'requires_supp' not in train_data_loader_config:
            train_data_loader_config['requires_supp'] = True

        val_data_loader_config = train_config.get('val_data_loader', dict())
        train_data_loader, val_data_loader =\
            build_data_loaders(self.dataset_dict, [train_data_loader_config, val_data_loader_config],
                               self.distributed, self.accelerator)
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
            auxiliary_teacher_model_wrapper = \
                build_auxiliary_model_wrapper(teacher_config, teacher_model=unwrapped_org_teacher_model,
                                              device=self.device, device_ids=self.device_ids,
                                              distributed=self.distributed)
            if auxiliary_teacher_model_wrapper is not None:
                teacher_ref_model = auxiliary_teacher_model_wrapper
                model_type = type(teacher_ref_model).__name__
            self.teacher_model = redesign_model(teacher_ref_model, teacher_config, 'teacher', model_type)

        if len(student_config) > 0 or (len(student_config) == 0 and self.student_model is None):
            model_type = 'original'
            auxiliary_student_model_wrapper = \
                build_auxiliary_model_wrapper(student_config, student_model=unwrapped_org_student_model,
                                              device=self.device, device_ids=self.device_ids,
                                              distributed=self.distributed)
            if auxiliary_student_model_wrapper is not None:
                student_ref_model = auxiliary_student_model_wrapper
                model_type = type(student_ref_model).__name__
            self.student_model = redesign_model(student_ref_model, student_config, 'student', model_type)

        self.teacher_any_frozen = \
            len(teacher_config.get('frozen_modules', list())) > 0 or not teacher_config.get('requires_grad', True)
        self.student_any_frozen = \
            len(student_config.get('frozen_modules', list())) > 0 or not student_config.get('requires_grad', True)
        self.target_teacher_pairs.extend(set_hooks(self.teacher_model, teacher_ref_model,
                                                   teacher_config, self.teacher_io_dict))
        self.target_student_pairs.extend(set_hooks(self.student_model, student_ref_model,
                                                   student_config, self.student_io_dict))
        self.teacher_forward_proc = get_forward_proc_func(teacher_config.get('forward_proc', None))
        self.student_forward_proc = get_forward_proc_func(student_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        self.criterion = get_high_level_loss(criterion_config)
        logger.info(self.criterion)
        self.extract_model_loss = get_func2extract_model_output(criterion_config.get('func2extract_model_loss', None))

    def setup_pre_post_processes(self, train_config):
        pre_epoch_process = default_pre_epoch_process_with_teacher
        if 'pre_epoch_process' in train_config:
            pre_epoch_process = get_pre_epoch_proc_func(train_config['pre_epoch_process'])
        setattr(DistillationBox, 'pre_epoch_process', pre_epoch_process)
        pre_forward_process = default_pre_forward_process
        if 'pre_forward_process' in train_config:
            pre_forward_process = get_pre_forward_proc_func(train_config['pre_forward_process'])
        setattr(DistillationBox, 'pre_forward_process', pre_forward_process)
        post_forward_process = default_post_forward_process
        if 'post_forward_process' in train_config:
            post_forward_process = get_post_forward_proc_func(train_config['post_forward_process'])

        setattr(DistillationBox, 'post_forward_process', post_forward_process)
        post_epoch_process = default_post_epoch_process_with_teacher
        if 'post_epoch_process' in train_config:
            post_epoch_process = get_post_epoch_proc_func(train_config['post_epoch_process'])
        setattr(DistillationBox, 'post_epoch_process', post_epoch_process)

    def setup(self, train_config):
        # Set up train and val data loaders
        self.setup_data_loaders(train_config)

        # Define teacher and student models used in this stage
        teacher_config = train_config.get('teacher', dict())
        student_config = train_config.get('student', dict())
        self.setup_teacher_student_models(teacher_config, student_config)

        # Define loss function used in this stage
        self.setup_loss(train_config)

        # Freeze parameters if specified
        self.teacher_updatable = True
        if not teacher_config.get('requires_grad', True):
            logger.info('Freezing the whole teacher model')
            freeze_module_params(self.teacher_model)
            self.teacher_updatable = False

        if not student_config.get('requires_grad', True):
            logger.info('Freezing the whole student model')
            freeze_module_params(self.student_model)

        # Wrap models if necessary
        teacher_unused_parameters = teacher_config.get('find_unused_parameters', self.teacher_any_frozen)
        teacher_any_updatable = len(get_updatable_param_names(self.teacher_model)) > 0
        self.teacher_model =\
            wrap_model(self.teacher_model, teacher_config, self.device, self.device_ids, self.distributed,
                       teacher_unused_parameters, teacher_any_updatable)
        student_unused_parameters = student_config.get('find_unused_parameters', self.student_any_frozen)
        student_any_updatable = len(get_updatable_param_names(self.student_model)) > 0
        self.student_model =\
            wrap_model(self.student_model, student_config, self.device, self.device_ids, self.distributed,
                       student_unused_parameters, student_any_updatable)

        # Set up optimizer and scheduler
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_kwargs = optim_config['kwargs']
            if 'lr' in optim_kwargs:
                optim_kwargs['lr'] *= self.lr_factor

            module_wise_kwargs_configs = optim_config.get('module_wise_kwargs', list())
            if len(module_wise_kwargs_configs) > 0:
                trainable_module_list = list()
                for module_wise_kwargs_config in module_wise_kwargs_configs:
                    module_wise_kwargs = dict()
                    if isinstance(module_wise_kwargs_config.get('kwargs', None), dict):
                        module_wise_kwargs.update(module_wise_kwargs_config['kwargs'])

                    if 'lr' in module_wise_kwargs:
                        module_wise_kwargs['lr'] *= self.lr_factor

                    target_model = \
                        self.teacher_model if module_wise_kwargs_config.get('is_teacher', False) else self.student_model
                    module = get_module(target_model, module_wise_kwargs_config['module'])
                    module_wise_kwargs['params'] = module.parameters()
                    trainable_module_list.append(module_wise_kwargs)
            else:
                trainable_module_list = nn.ModuleList([self.student_model])
                if self.teacher_updatable:
                    logger.info('Note that you are training some/all of the modules in the teacher model')
                    trainable_module_list.append(self.teacher_model)

            filters_params = optim_config.get('filters_params', True)
            self.optimizer = \
                get_optimizer(trainable_module_list, optim_config['type'],
                              **optim_kwargs, filters_params=filters_params)

            self.optimizer.zero_grad()
            self.max_grad_norm = optim_config.get('max_grad_norm', None)
            self.grad_accum_step = optim_config.get('grad_accum_step', 1)
            optimizer_reset = True

        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['type'], **scheduler_config['kwargs'])
            self.scheduling_step = scheduler_config.get('scheduling_step', 0)
        elif optimizer_reset:
            self.lr_scheduler = None
            self.scheduling_step = None

        # Set up accelerator if necessary
        if self.accelerator is not None:
            if self.teacher_updatable:
                self.teacher_model, self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader = \
                    self.accelerator.prepare(self.teacher_model, self.student_model, self.optimizer,
                                             self.train_data_loader, self.val_data_loader)
            else:
                self.teacher_model = self.teacher_model.to(self.accelerator.device)
                if self.accelerator.state.use_fp16:
                    self.teacher_model = self.teacher_model.half()

                self.student_model, self.optimizer, self.train_data_loader, self.val_data_loader = \
                    self.accelerator.prepare(self.student_model, self.optimizer,
                                             self.train_data_loader, self.val_data_loader)

        # Set up {pre,post}-{epoch,forward} processes
        self.setup_pre_post_processes(train_config)


    def __init__(self, teacher_model, student_model, dataset_dict,
                 train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        super().__init__()
        # Key attributes (should not be modified)
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
        self.accelerator = accelerator
        # Local attributes (can be updated at each stage)
        self.teacher_model = None
        self.student_model = None
        self.teacher_forward_proc, self.student_forward_proc = None, None
        self.target_teacher_pairs, self.target_student_pairs = list(), list()
        self.teacher_io_dict, self.student_io_dict = dict(), dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.criterion, self.extract_model_loss = None, None
        self.teacher_updatable, self.teacher_any_frozen, self.student_any_frozen = None, None, None
        self.grad_accum_step = None
        self.max_grad_norm = None
        self.scheduling_step = 0
        self.stage_grad_count = 0
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_epoch_process(self, *args, **kwargs):
        raise NotImplementedError()

    def get_teacher_output(self, sample_batch, targets, supp_dict):
        if supp_dict is None:
            supp_dict = dict()

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

        # If no cached data
        if teacher_outputs is None:
            if self.teacher_updatable:
                teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)
            else:
                with torch.no_grad():
                    teacher_outputs = self.teacher_forward_proc(self.teacher_model, sample_batch, targets, supp_dict)

        if cached_extracted_teacher_output_dict is not None:
            if isinstance(self.teacher_model, AuxiliaryModelWrapper) or \
                    (check_if_wrapped(self.teacher_model) and isinstance(self.teacher_model.module, AuxiliaryModelWrapper)):
                self.teacher_io_dict.update(cached_extracted_teacher_output_dict)
                if isinstance(self.teacher_model, AuxiliaryModelWrapper):
                    self.teacher_model.secondary_forward(self.teacher_io_dict)

            extracted_teacher_io_dict = extract_io_dict(self.teacher_io_dict, self.device)
            return teacher_outputs, extracted_teacher_io_dict

        # Deep copy of teacher info dict if auxiliary teacher model wrapper contains trainable module(s)
        teacher_io_dict4cache = copy.deepcopy(self.teacher_io_dict) \
            if self.teacher_updatable and isinstance(cache_file_paths, (list, tuple)) is not None else None
        extracted_teacher_io_dict = extract_io_dict(self.teacher_io_dict, self.device)
        extracted_teacher_io_dict[SELF_MODULE_PATH]['output'] = teacher_outputs
        if isinstance(self.teacher_model, AuxiliaryModelWrapper):
            self.teacher_model.secondary_forward(extracted_teacher_io_dict)

        update_io_dict(extracted_teacher_io_dict, extract_io_dict(self.teacher_io_dict, self.device))
        # Write cache files if output file paths (cache_file_paths) are given
        if isinstance(cache_file_paths, (list, tuple)):
            if teacher_io_dict4cache is None:
                teacher_io_dict4cache = extracted_teacher_io_dict

            cpu_device = torch.device('cpu')
            for i, (teacher_output, cache_file_path) in enumerate(zip(teacher_outputs.cpu().numpy(), cache_file_paths)):
                sub_dict = extract_sub_model_output_dict(teacher_io_dict4cache, i)
                sub_dict = tensor2numpy2tensor(sub_dict, cpu_device)
                cache_dict = {'teacher_outputs': torch.Tensor(teacher_output), 'extracted_outputs': sub_dict}
                make_parent_dirs(cache_file_path)
                torch.save(cache_dict, cache_file_path)
        return teacher_outputs, extracted_teacher_io_dict

    def forward(self, sample_batch, targets, supp_dict):
        teacher_outputs, extracted_teacher_io_dict =\
            self.get_teacher_output(sample_batch, targets, supp_dict=supp_dict)
        student_outputs = self.student_forward_proc(self.student_model, sample_batch, targets, supp_dict)
        extracted_student_io_dict = extract_io_dict(self.student_io_dict, self.device)
        extracted_student_io_dict[SELF_MODULE_PATH]['output'] = student_outputs
        if isinstance(self.student_model, AuxiliaryModelWrapper):
            self.student_model.secondary_forward(extracted_student_io_dict)

        model_loss_dict = self.extract_model_loss(student_outputs, targets, supp_dict=supp_dict)
        update_io_dict(extracted_student_io_dict, extract_io_dict(self.student_io_dict, self.device))
        io_dict = {'teacher': extracted_teacher_io_dict, 'student': extracted_student_io_dict}
        total_loss = self.criterion(io_dict, model_loss_dict, targets)
        return total_loss

    def post_forward_process(self, *args, **kwargs):
        raise NotImplementedError()

    def post_epoch_process(self, *args, **kwargs):
        raise NotImplementedError()

    def clean_modules(self):
        unfreeze_module_params(self.org_teacher_model)
        unfreeze_module_params(self.org_student_model)
        self.teacher_io_dict.clear()
        self.student_io_dict.clear()
        for _, module_handle in self.target_teacher_pairs + self.target_student_pairs:
            module_handle.remove()

        self.target_teacher_pairs.clear()
        self.target_student_pairs.clear()


class MultiStagesDistillationBox(DistillationBox):
    def __init__(self, teacher_model, student_model, data_loader_dict,
                 train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        stage1_config = train_config['stage1']
        super().__init__(teacher_model, student_model, data_loader_dict,
                         stage1_config, device, device_ids, distributed, lr_factor, accelerator)
        self.train_config = train_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['num_epochs']
        self.num_epochs = sum(train_config[key]['num_epochs'] for key in train_config.keys() if key.startswith('stage'))
        self.current_epoch = 0
        logger.info('Started stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        self.clean_modules()
        self.stage_grad_count = 0
        self.stage_number += 1
        next_stage_config = self.train_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config)
        self.stage_end_epoch += next_stage_config['num_epochs']
        logger.info('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, **kwargs):
        super().post_process(**kwargs)
        self.current_epoch += 1
        if self.current_epoch == self.stage_end_epoch and self.current_epoch < self.num_epochs:
            self.advance_to_next_stage()


def get_distillation_box(teacher_model, student_model, data_loader_dict,
                         train_config, device, device_ids, distributed, lr_factor, accelerator=None):
    if 'stage1' in train_config:
        return MultiStagesDistillationBox(teacher_model, student_model, data_loader_dict,
                                          train_config, device, device_ids, distributed, lr_factor, accelerator)
    return DistillationBox(teacher_model, student_model, data_loader_dict, train_config,
                           device, device_ids, distributed, lr_factor, accelerator)

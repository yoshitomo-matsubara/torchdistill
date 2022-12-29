import torch
from torch import distributed as dist
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from .post_epoch_proc import default_post_epoch_process_with_teacher
from .post_forward_proc import default_post_forward_process
from .pre_epoch_proc import default_pre_epoch_process_without_teacher
from .pre_forward_proc import default_pre_forward_process
from .registry import get_pre_epoch_proc_func, get_pre_forward_proc_func, get_forward_proc_func, \
    get_post_forward_proc_func, get_post_epoch_proc_func
from .util import set_hooks, wrap_model, clear_io_dict, extract_io_dict, update_io_dict
from ..common.constant import SELF_MODULE_PATH, def_logger
from ..common.module_util import check_if_wrapped, freeze_module_params, get_module, \
    unfreeze_module_params, get_updatable_param_names
from ..datasets.util import build_data_loaders
from ..losses.registry import get_custom_loss, get_single_loss, get_func2extract_org_output
from ..models.special import AuxiliaryModelWrapper, build_auxiliary_model_wrapper
from ..models.util import redesign_model
from ..optim.registry import get_optimizer, get_scheduler

logger = def_logger.getChild(__name__)


class TrainingBox(nn.Module):
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

    def setup_model(self, model_config):
        unwrapped_org_model = \
            self.org_model.module if check_if_wrapped(self.org_model) else self.org_model
        self.target_model_pairs.clear()
        ref_model = unwrapped_org_model

        if len(model_config) > 0 or (len(model_config) == 0 and self.model is None):
            model_type = 'original'
            special_model = \
                build_auxiliary_model_wrapper(model_config, student_model=unwrapped_org_model, device=self.device,
                                     device_ids=self.device_ids, distributed=self.distributed)
            if special_model is not None:
                ref_model = special_model
                model_type = type(ref_model).__name__
            self.model = redesign_model(ref_model, model_config, 'student', model_type)

        self.model_any_frozen = \
            len(model_config.get('frozen_modules', list())) > 0 or not model_config.get('requires_grad', True)
        self.target_model_pairs.extend(set_hooks(self.model, ref_model, model_config, self.model_io_dict))
        self.model_forward_proc = get_forward_proc_func(model_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        self.org_criterion = None if org_criterion_config is None or len(org_criterion_config) == 0 \
            else get_single_loss(org_criterion_config)
        self.criterion = get_custom_loss(criterion_config)
        logger.info(self.criterion)
        self.extract_org_loss = get_func2extract_org_output(criterion_config.get('func2extract_org_loss', None))

    def setup_pre_post_processes(self, train_config):
        pre_epoch_process = default_pre_epoch_process_without_teacher
        if 'pre_epoch_process' in train_config:
            pre_epoch_process = get_pre_epoch_proc_func(train_config['pre_epoch_process'])
        setattr(TrainingBox, 'pre_epoch_process', pre_epoch_process)
        pre_forward_process = default_pre_forward_process
        if 'pre_forward_process' in train_config:
            pre_forward_process = get_pre_forward_proc_func(train_config['pre_forward_process'])
        setattr(TrainingBox, 'pre_forward_process', pre_forward_process)
        post_forward_process = default_post_forward_process
        if 'post_forward_process' in train_config:
            post_forward_process = get_post_forward_proc_func(train_config['post_forward_process'])

        setattr(TrainingBox, 'post_forward_process', post_forward_process)
        post_epoch_process = default_post_epoch_process_with_teacher
        if 'post_epoch_process' in train_config:
            post_epoch_process = get_post_epoch_proc_func(train_config['post_epoch_process'])
        setattr(TrainingBox, 'post_epoch_process', post_epoch_process)

    def setup(self, train_config):
        # Set up train and val data loaders
        self.setup_data_loaders(train_config)

        # Define model used in this stage
        model_config = train_config.get('model', dict())
        self.setup_model(model_config)

        # Define loss function used in this stage
        self.setup_loss(train_config)

        # Freeze parameters if specified
        if not model_config.get('requires_grad', True):
            logger.info('Freezing the whole model')
            freeze_module_params(self.model)

        # Wrap models if necessary
        any_updatable = len(get_updatable_param_names(self.model)) > 0
        model_unused_parameters = model_config.get('find_unused_parameters', self.model_any_frozen)
        self.model =\
            wrap_model(self.model, model_config, self.device, self.device_ids, self.distributed,
                       model_unused_parameters, any_updatable)

        # Set up optimizer and scheduler
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_params_config = optim_config['params']
            if 'lr' in optim_params_config:
                optim_params_config['lr'] *= self.lr_factor

            module_wise_params_configs = optim_config.get('module_wise_params', list())
            if len(module_wise_params_configs) > 0:
                trainable_module_list = list()
                for module_wise_params_config in module_wise_params_configs:
                    module_wise_params_dict = dict()
                    if isinstance(module_wise_params_config.get('params', None), dict):
                        module_wise_params_dict.update(module_wise_params_config['params'])

                    if 'lr' in module_wise_params_dict:
                        module_wise_params_dict['lr'] *= self.lr_factor

                    module = get_module(self.model, module_wise_params_config['module'])
                    module_wise_params_dict['params'] = module.parameters()
                    trainable_module_list.append(module_wise_params_dict)
            else:
                trainable_module_list = nn.ModuleList([self.model])

            filters_params = optim_config.get('filters_params', True)
            self.optimizer = \
                get_optimizer(trainable_module_list, optim_config['type'], optim_params_config, filters_params)
            self.optimizer.zero_grad()
            self.max_grad_norm = optim_config.get('max_grad_norm', None)
            self.grad_accum_step = optim_config.get('grad_accum_step', 1)
            optimizer_reset = True

        scheduler_config = train_config.get('scheduler', None)
        if scheduler_config is not None and len(scheduler_config) > 0:
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
            self.scheduling_step = scheduler_config.get('scheduling_step', 0)
        elif optimizer_reset:
            self.lr_scheduler = None
            self.scheduling_step = None

        # Set up accelerator if necessary
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_data_loader, self.val_data_loader = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_data_loader, self.val_data_loader)

        # Set up {pre,post}-{epoch,forward} processes
        self.setup_pre_post_processes(train_config)

    def __init__(self, model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
        super().__init__()
        # Key attributes (should not be modified)
        self.org_model = model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
        self.accelerator = accelerator
        # Local attributes (can be updated at each stage)
        self.model = None
        self.model_forward_proc = None
        self.target_model_pairs = list()
        self.model_io_dict = dict()
        self.train_data_loader, self.val_data_loader, self.optimizer, self.lr_scheduler = None, None, None, None
        self.org_criterion, self.criterion, self.extract_org_loss = None, None, None
        self.model_any_frozen = None
        self.grad_accum_step = None
        self.max_grad_norm = None
        self.scheduling_step = 0
        self.stage_grad_count = 0
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_process(self, epoch=None, **kwargs):
        clear_io_dict(self.model_io_dict)
        self.model.train()
        if self.distributed:
            self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)

    def forward(self, sample_batch, targets, supp_dict):
        model_outputs = self.model_forward_proc(self.model, sample_batch, targets, supp_dict)
        extracted_model_io_dict = extract_io_dict(self.model_io_dict, self.device)
        extracted_model_io_dict[SELF_MODULE_PATH]['output'] = model_outputs
        if isinstance(self.model, AuxiliaryModelWrapper):
            self.model.secondary_forward(extracted_model_io_dict)

        org_loss_dict = self.extract_org_loss(self.org_criterion, model_outputs, targets,
                                              supp_dict=supp_dict)
        update_io_dict(extracted_model_io_dict, extract_io_dict(self.model_io_dict, self.device))
        io_dict = {'student': extracted_model_io_dict, 'teacher': dict()}
        total_loss = self.criterion(io_dict, org_loss_dict, targets)
        return total_loss

    def update_params(self, loss, **kwargs):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = [p for group in self.optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        # Step-wise scheduler step
        if self.lr_scheduler is not None and self.scheduling_step > 0 \
                and self.stage_grad_count % self.scheduling_step == 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                local_epoch = int(self.stage_grad_count / self.scheduling_step)
                self.lr_scheduler.step(local_epoch)
            else:
                self.lr_scheduler.step()

    def post_process(self, **kwargs):
        # Epoch-wise scheduler step
        if self.lr_scheduler is not None and self.scheduling_step <= 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                metrics = kwargs['metrics']
                self.lr_scheduler.step(metrics)
            elif isinstance(self.lr_scheduler, LambdaLR):
                epoch = self.lr_scheduler.last_epoch + 1
                self.lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step()
        if isinstance(self.model, AuxiliaryModelWrapper):
            self.model.post_process()
        if self.distributed:
            dist.barrier()

    def clean_modules(self):
        unfreeze_module_params(self.org_model)
        self.model_io_dict.clear()
        for _, module_handle in self.target_model_pairs:
            module_handle.remove()
        self.target_model_pairs.clear()


class MultiStagesTrainingBox(TrainingBox):
    def __init__(self, model, data_loader_dict, train_config,
                 device, device_ids, distributed, lr_factor, accelerator=None):
        stage1_config = train_config['stage1']
        super().__init__(model, data_loader_dict,
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


def get_training_box(model, data_loader_dict, train_config, device, device_ids, distributed,
                     lr_factor, accelerator=None):
    if 'stage1' in train_config:
        return MultiStagesTrainingBox(model, data_loader_dict,
                                      train_config, device, device_ids, distributed, lr_factor, accelerator)
    return TrainingBox(model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor, accelerator)

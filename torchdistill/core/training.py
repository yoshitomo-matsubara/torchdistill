import sys

import torch
from torch import distributed as dist
from torch import nn

from torchdistill.common.constant import def_logger
from torchdistill.common.func_util import get_optimizer, get_scheduler
from torchdistill.common.module_util import check_if_wrapped, freeze_module_params, get_module, unfreeze_module_params
from torchdistill.core.forward_proc import get_forward_proc_func
from torchdistill.core.util import set_hooks, wrap_model, extract_io_dict, update_io_dict
from torchdistill.datasets.util import build_data_loaders
from torchdistill.losses.custom import get_custom_loss
from torchdistill.losses.single import get_single_loss
from torchdistill.losses.util import get_func2extract_org_output
from torchdistill.models.special import SpecialModule, build_special_module
from torchdistill.models.util import redesign_model

logger = def_logger.getChild(__name__)
try:
    from apex import amp
except ImportError:
    amp = None


class TrainingBox(nn.Module):
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

    def setup_model(self, model_config):
        unwrapped_org_model = \
            self.org_model.module if check_if_wrapped(self.org_model) else self.org_model
        self.target_model_pairs.clear()
        ref_model = unwrapped_org_model

        if len(model_config) > 0 or (len(model_config) == 0 and self.model is None):
            model_type = 'original'
            special_model = \
                build_special_module(model_config, student_model=unwrapped_org_model, device=self.device,
                                     device_ids=self.device_ids, distributed=self.distributed)
            if special_model is not None:
                ref_model = special_model
                model_type = type(ref_model).__name__
            self.model = redesign_model(ref_model, model_config, 'student', model_type)

        self.model_any_frozen = \
            len(model_config.get('frozen_modules', list())) > 0 or not model_config.get('requires_grad', True)
        self.target_model_pairs.extend(set_hooks(self.model, ref_model,
                                                   model_config, self.model_io_dict))
        self.model_forward_proc = get_forward_proc_func(model_config.get('forward_proc', None))

    def setup_loss(self, train_config):
        criterion_config = train_config['criterion']
        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        self.org_criterion = None if org_criterion_config is None or len(org_criterion_config) == 0 \
            else get_single_loss(org_criterion_config)
        self.criterion = get_custom_loss(criterion_config)
        self.uses_teacher_output = False
        self.extract_org_loss = get_func2extract_org_output(criterion_config.get('func2extract_org_loss', None))

    def setup(self, train_config):
        # Set up train and val data loaders
        self.setup_data_loaders(train_config)

        # Define model used in this stage
        model_config = train_config.get('model', dict())
        self.setup_model(model_config)

        # Define loss function used in this stage
        self.setup_loss(train_config)

        # Wrap models if necessary
        self.model =\
            wrap_model(self.model, model_config, self.device, self.device_ids, self.distributed,
                       self.model_any_frozen)

        if not model_config.get('requires_grad', True):
            logger.info('Freezing the whole model')
            freeze_module_params(self.model)

        # Set up optimizer and scheduler
        optim_config = train_config.get('optimizer', dict())
        optimizer_reset = False
        if len(optim_config) > 0:
            optim_params_config = optim_config['params']
            optim_params_config['lr'] *= self.lr_factor
            module_wise_params_configs = optim_config.get('module_wise_params', list())
            if len(module_wise_params_configs) > 0:
                trainable_module_list = list()
                for module_wise_params_config in module_wise_params_configs:
                    module_wise_params_dict = dict()
                    module_wise_params_dict.update(module_wise_params_config['params'])
                    if 'lr' in module_wise_params_dict:
                        module_wise_params_dict['lr'] *= self.lr_factor

                    module = get_module(self.model, module_wise_params_config['module'])
                    module_wise_params_dict['params'] = module.parameters()
                    trainable_module_list.append(module_wise_params_dict)
            else:
                trainable_module_list = nn.ModuleList([self.model])

            self.optimizer = get_optimizer(trainable_module_list, optim_config['type'], optim_params_config)
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

        # Set up apex if you require mixed-precision training
        self.apex = False
        apex_config = train_config.get('apex', None)
        if apex_config is not None and apex_config.get('requires', False):
            if sys.version_info < (3, 0):
                raise RuntimeError('Apex currently only supports Python 3. Aborting.')
            if amp is None:
                raise RuntimeError('Failed to import apex. Please install apex from https://www.github.com/nvidia/apex '
                                   'to enable mixed-precision training.')
            self.model, self.optimizer =\
                amp.initialize(self.model, self.optimizer, opt_level=apex_config['opt_level'])
            self.apex = True

    def __init__(self, model, dataset_dict, train_config, device, device_ids, distributed, lr_factor):
        super().__init__()
        # Key attributes (should not be modified)
        self.org_model = model
        self.dataset_dict = dataset_dict
        self.device = device
        self.device_ids = device_ids
        self.distributed = distributed
        self.lr_factor = lr_factor
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
        self.apex = None
        self.setup(train_config)
        self.num_epochs = train_config['num_epochs']

    def pre_process(self, epoch=None, **kwargs):
        self.model.train()
        if self.distributed:
            self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)

    def forward(self, sample_batch, targets, supp_dict):
        model_outputs = self.model_forward_proc(self.model, sample_batch, targets, supp_dict)
        extracted_model_io_dict = extract_io_dict(self.model_io_dict, self.device)
        if isinstance(self.model, SpecialModule):
            self.model.post_forward(extracted_model_io_dict)

        teacher_outputs = None
        org_loss_dict = self.extract_org_loss(self.org_criterion, model_outputs, teacher_outputs, targets,
                                              uses_teacher_output=False, supp_dict=supp_dict)
        update_io_dict(extracted_model_io_dict, extract_io_dict(self.model_io_dict, self.device))
        output_dict = {'student': extracted_model_io_dict, 'teacher': dict()}
        total_loss = self.criterion(output_dict, org_loss_dict, targets)
        return total_loss

    def update_params(self, loss):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step

        if self.apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = amp.master_params(self.optimizer) if self.apex \
                    else [p for group in self.optimizer.param_groups for p in group['group']]
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        # Step-wise scheduler step
        if self.lr_scheduler is not None and self.scheduling_step > 0 \
                and self.stage_grad_count % self.scheduling_step == 0:
            self.lr_scheduler.step()

    def post_process(self, **kwargs):
        # Epoch-wise scheduler step
        if self.lr_scheduler is not None and self.scheduling_step <= 0:
            self.lr_scheduler.step()
        if isinstance(self.model, SpecialModule):
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
    def __init__(self, model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor):
        stage1_config = train_config['stage1']
        super().__init__(model, data_loader_dict,
                         stage1_config, device, device_ids, distributed, lr_factor)
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
        super().post_process()
        self.current_epoch += 1
        if self.current_epoch == self.stage_end_epoch and self.current_epoch < self.num_epochs:
            self.advance_to_next_stage()


def get_training_box(model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor):
    if 'stage1' in train_config:
        return MultiStagesTrainingBox(model, data_loader_dict,
                                      train_config, device, device_ids, distributed, lr_factor)
    return TrainingBox(model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor)

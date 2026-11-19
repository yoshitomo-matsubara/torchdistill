import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, Adafactor

from torchdistill.optim.registry import register_optimizer, register_scheduler, get_optimizer

# Register lr schedulers
for func in TYPE_TO_SCHEDULER_FUNCTION.values():
    register_scheduler(func)

# T5's Adafactor, under its own key: torch.optim.Adafactor is already registered as 'Adafactor' and lacks the
# scale_parameter / relative_step options
@register_optimizer(key='HFAdafactor')
class DTensorAwareAdafactor(Adafactor):
    # Under FSDP2, parameters and gradients are DTensors sharded on dim 0. Adafactor creates its factored second
    # moments as plain tensors, which DTensor ops reject, so they are created here as DTensors first: the row
    # statistics keep the gradient's sharding, and the column statistics, which reduce over the sharded dim, are
    # replicated. Everything else is Adafactor's own step, unchanged for plain tensors
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                if grad is None or not isinstance(grad, DTensor) or len(self.state[p]) > 0:
                    continue

                factored, use_first_moment = self._get_options(group, grad.shape)
                if not factored:
                    continue

                grad = grad.float()
                state = self.state[p]
                state['step'] = 0
                if use_first_moment:
                    state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq_row'] = torch.zeros_like(grad.mean(dim=-1))
                state['exp_avg_sq_col'] = torch.zeros_like(
                    grad.mean(dim=-2).redistribute(grad.device_mesh, [Replicate()] * grad.device_mesh.ndim)
                )
                state['RMS'] = 0
        return super().step(closure)


def compute_num_training_steps(dataset, batch_size, num_epochs, grad_accum_step=1, world_size=1):
    num_iterations = math.ceil(len(dataset) / batch_size / grad_accum_step / world_size)
    return num_iterations * num_epochs


def update_num_iterations(train_config, dataset_dict, world_size):
    if 'scheduler' not in train_config or 'train_data_loader' not in train_config:
        return

    train_data_loader_config = train_config['train_data_loader']
    grad_accum_step = train_config.get('grad_accum_step', 1)
    num_iterations = math.ceil(
        len(dataset_dict[train_data_loader_config['dataset_id']]) / train_data_loader_config['kwargs']['batch_size']
        / grad_accum_step / world_size
    )
    scheduler_config = train_config['scheduler']
    scheduler_config['kwargs']['num_training_steps'] = num_iterations * train_config['num_epochs']


def customize_lr_config(config, dataset_dict, world_size):
    if 'train' not in config:
        return

    train_config = config['train']
    if 'stage1' not in train_config:
        update_num_iterations(train_config, dataset_dict, world_size)
    else:
        for i in range(len(train_config.keys())):
            stage_name = 'stage{}'.format(i + 1)
            if stage_name not in train_config:
                break

            stage_train_config = train_config[stage_name]
            update_num_iterations(stage_train_config, dataset_dict, world_size)


@register_optimizer
def optimizer_no_decay(model, optimizer_key, weight_decay, no_decay=None, **kwargs):
    if no_decay is None:
        no_decay = ['bias', 'LayerNorm.weight']

    if isinstance(model, nn.Module):
        model = [(n, p) for n, p in model.named_parameters()]

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = get_optimizer(optimizer_grouped_parameters, optimizer_key, **kwargs)
    return optimizer

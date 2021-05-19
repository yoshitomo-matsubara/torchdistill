import math

from torch import nn
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from torchdistill.optim.registry import register_optimizer, register_scheduler, get_optimizer

# Register lr schedulers
for func in TYPE_TO_SCHEDULER_FUNCTION.values():
    register_scheduler(func)


def update_num_iterations(train_config, dataset_dict, world_size):
    if 'scheduler' not in train_config or 'train_data_loader' not in train_config:
        return

    train_data_loader_config = train_config['train_data_loader']
    grad_accum_step = train_config.get('grad_accum_step', 1)
    num_iterations = math.ceil(len(dataset_dict[train_data_loader_config['dataset_id']]) /
                               train_data_loader_config['batch_size'] / grad_accum_step / world_size)
    scheduler_config = train_config['scheduler']
    scheduler_config['params']['num_training_steps'] = num_iterations * train_config['num_epochs']


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
def optimizer_no_decay(model, optimizer_type, weight_decay, no_decay=None, **kwargs):
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
    optimizer = get_optimizer(optimizer_grouped_parameters, optimizer_type, **kwargs)
    return optimizer

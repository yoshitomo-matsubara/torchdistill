import math


def update_num_iterations(train_config, dataset_dict, world_size):
    if 'scheduler' not in train_config or 'train_data_loader' not in train_config:
        return

    train_data_loader_config = train_config['train_data_loader']
    num_iterations = math.ceil(len(dataset_dict[train_data_loader_config['dataset_id']]) /
                               train_data_loader_config['batch_size'] / world_size)
    scheduler_config = train_config['scheduler']
    scheduler_type = scheduler_config['type']
    if scheduler_type == 'poly_lr_scheduler':
        scheduler_config['kwargs']['num_iterations'] = num_iterations


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

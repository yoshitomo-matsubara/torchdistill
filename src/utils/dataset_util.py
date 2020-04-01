import hashlib
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from myutils.common import file_util
from utils import main_util


def get_cache_path(file_path, dataset_type):
    h = hashlib.sha1(file_path.encode()).hexdigest()
    cache_path = os.path.join('~', '.torch', 'vision', 'datasets', dataset_type, h[:10] + '.pt')
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_image_folder_dataset(dir_path, data_aug, dataset_type, rough_size, input_size,
                              normalizer, use_cache, split_name):
    input_size = tuple(input_size)
    # Data loading code
    st = time.time()
    cache_path = get_cache_path(dir_path, dataset_type)
    if data_aug:
        if use_cache and file_util.check_if_exists(cache_path):
            # Attention, as the transforms are also cached!
            print('Loading cached training dataset from {}'.format(cache_path))
            train_dataset, _ = torch.load(cache_path)
            return train_dataset
        else:
            print('Loading {} data'.format(split_name))
            train_dataset = ImageFolder(
                dir_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalizer,
                ]))
            if use_cache:
                print('Saving dataset_train to {}'.format(cache_path))
                file_util.make_parent_dirs(cache_path)
                main_util.save_on_master((train_dataset, dir_path), cache_path)
            print('\t', time.time() - st)
            return train_dataset

    if use_cache and file_util.check_if_exists(cache_path):
        # Attention, as the transforms are also cached!
        print('Loading cached {} dataset from {}'.format(split_name, cache_path))
        eval_dataset, _ = torch.load(cache_path)
        return eval_dataset

    print('Loading {} data'.format(split_name))
    eval_dataset = ImageFolder(
        dir_path,
        transforms.Compose([
            transforms.Resize(rough_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalizer,
        ]))
    if use_cache:
        print('Saving {} dataset_test to {}'.format(split_name, cache_path))
        file_util.make_parent_dirs(cache_path)
        main_util.save_on_master((eval_dataset, dir_path), cache_path)

    print('\t', time.time() - st)
    return eval_dataset


def get_dataset_dict(dataset_config, use_cache):
    dataset_type = dataset_config['type']
    rough_size = dataset_config['rough_size']
    input_size = dataset_config['input_size']
    normalizer = transforms.Normalize(**dataset_config['normalizer'])
    dataset_dict = dict()
    if dataset_type == 'imagefolder':
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            split_config = dataset_splits_config[split_name]
            dataset_dict[split_config['dataset_id']] =\
                load_image_folder_dataset(split_config['images'], split_config['data_aug'], dataset_type, rough_size,
                                          input_size, normalizer, use_cache, split_name)
    else:
        raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))
    return dataset_dict


def get_all_dataset(datasets_config, use_cache):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_dataset_dict(datasets_config[dataset_name], use_cache)
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict


def build_data_loader(dataset, data_loader_config, distributed):
    batch_size, num_workers = data_loader_config['batch_size'], data_loader_config['num_workers']
    sampler = DistributedSampler(dataset) if distributed \
        else RandomSampler(dataset) if data_loader_config.get('random_sample', False) else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


def build_data_loaders(dataset_dict, data_loader_configs, distributed):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed)
        data_loader_list.append(data_loader)
    return data_loader_list

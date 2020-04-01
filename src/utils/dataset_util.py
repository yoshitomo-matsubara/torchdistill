import hashlib
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from myutils.common import file_util
from utils import main_util


def get_cache_path(file_path, dataset_type):
    h = hashlib.sha1(file_path.encode()).hexdigest()
    cache_path = os.path.join('~', '.torch', 'vision', 'datasets', dataset_type, h[:10] + '.pt')
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_image_folder_dataset(dir_path, dataset_type, rough_size, input_size, normalizer, use_cache, split_name):
    input_size = tuple(input_size)
    # Data loading code
    st = time.time()
    cache_path = get_cache_path(dir_path, dataset_type)
    if split_name == 'train':
        if use_cache and file_util.check_if_exists(cache_path):
            # Attention, as the transforms are also cached!
            print('Loading cached training dataset from {}'.format(cache_path))
            train_dataset, _ = torch.load(cache_path)
            return train_dataset
        else:
            print('Loading training data')
            train_dataset = torchvision.datasets.ImageFolder(
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
    eval_dataset = torchvision.datasets.ImageFolder(
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


def get_data_loaders(dataset_config, distill_batch_size, test_batch_size, use_cache, distributed):
    dataset_type = dataset_config['type']
    rough_size = dataset_config['rough_size']
    input_size = dataset_config['input_size']
    normalizer = transforms.Normalize(**dataset_config['normalizer'])
    train_dict = dict()
    if dataset_type == 'imagefolder':
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            if split_name not in ('val', 'test'):
                train_dict[split_name] = dict()
                train_dataset = load_image_folder_dataset(dataset_splits_config[split_name]['images'], dataset_type,
                                                          rough_size, input_size, normalizer, use_cache, 'train')
                train_dict[split_name]['dataset'] = train_dataset

        val_dir_path = dataset_splits_config['val']['images']
        val_dataset = load_image_folder_dataset(val_dir_path, dataset_type,
                                                rough_size, input_size, normalizer, use_cache, 'validation')
        test_dir_path = dataset_splits_config['test']['images']
        if test_dir_path is not None and test_dir_path != val_dir_path:
            test_dataset = load_image_folder_dataset(test_dir_path, dataset_type, rough_size, input_size, normalizer,
                                                     use_cache, 'test')
        else:
            print('Shallow-copying validation dataset for test dataset')
            test_dataset = val_dataset
    else:
        raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))

    if distributed:
        for train_key in train_dict.keys():
            train_dict[train_key]['sampler'] = DistributedSampler(train_dict[train_key]['dataset'])

        val_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)
    else:
        for train_key in train_dict.keys():
            train_dict[train_key]['sampler'] = RandomSampler(train_dict[train_key]['dataset'])

        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)

    data_loader_dict = dict()
    num_workers = dataset_config['num_workers']
    for train_key in train_dict.keys():
        data_loader_dict[train_key] =\
            DataLoader(train_dict[train_key]['dataset'], batch_size=distill_batch_size,
                       sampler=train_dict[train_key]['sampler'], num_workers=num_workers, pin_memory=True)

    data_loader_dict['val'] = DataLoader(val_dataset, batch_size=distill_batch_size, sampler=val_sampler,
                                         num_workers=num_workers, pin_memory=True)
    data_loader_dict['test'] = DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler,
                                          num_workers=num_workers, pin_memory=True)
    return data_loader_dict

import copy
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from ..common.constant import def_logger
from ..datasets.registry import get_dataset, get_collate_func, get_batch_sampler, get_dataset_wrapper
from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def split_dataset(org_dataset, random_split_config, dataset_id, dataset_dict):
    org_dataset_length = len(org_dataset)
    logger.info('Splitting `{}` dataset ({} samples in total)'.format(dataset_id, org_dataset_length))
    lengths = random_split_config['lengths']
    total_length = sum(lengths)
    if total_length != org_dataset_length:
        lengths = [int((l / total_length) * org_dataset_length) for l in lengths]
        if len(lengths) > 1 and sum(lengths) != org_dataset_length:
            lengths[-1] = org_dataset_length - sum(lengths[:-1])

    manual_seed = random_split_config.get('generator_seed', None)
    sub_datasets = random_split(org_dataset, lengths) if manual_seed is None \
        else random_split(org_dataset, lengths, generator=torch.Generator().manual_seed(manual_seed))
    # Deep-copy dataset to configure transforms independently as dataset in Subset class is shallow-copied
    for sub_dataset in sub_datasets:
        sub_dataset.dataset = copy.deepcopy(sub_dataset.dataset)

    sub_splits_config = random_split_config['sub_splits']
    assert len(sub_datasets) == len(sub_splits_config), \
        'len(lengths) `{}` should be equal to len(sub_splits) `{}`'.format(len(sub_datasets), len(sub_splits_config))
    for sub_dataset, sub_split_kwargs in zip(sub_datasets, sub_splits_config):
        sub_dataset_id = sub_split_kwargs['dataset_id']
        logger.info('new dataset_id: `{}` ({} samples)'.format(sub_dataset_id, len(sub_dataset)))
        sub_split_kwargs = sub_split_kwargs.copy()
        transform = sub_split_kwargs.pop('transform', None)
        target_transform = sub_split_kwargs.pop('target_transform', None)
        transforms = sub_split_kwargs.pop('transforms', None)
        if hasattr(sub_dataset.dataset, 'transform') and transform is not None:
            sub_dataset.dataset.transform = transform
        if hasattr(sub_dataset.dataset, 'target_transform') and target_transform is not None:
            sub_dataset.dataset.target_transform = target_transform
        if hasattr(sub_dataset.dataset, 'transforms') and transforms is not None:
            sub_dataset.dataset.transforms = transforms
        dataset_dict[sub_dataset_id] = sub_dataset


def get_dataset_dict(dataset_config):
    dataset_key = dataset_config['key']
    dataset_dict = dict()
    dataset_cls_or_func = get_dataset(dataset_key)
    dataset_splits_config = dataset_config['splits']
    for split_name in dataset_splits_config.keys():
        st = time.time()
        logger.info('Loading {} data'.format(split_name))
        split_config = dataset_splits_config[split_name]
        org_dataset = dataset_cls_or_func(**split_config['kwargs'])
        dataset_id = split_config['dataset_id']
        random_split_config = split_config.get('random_split', None)
        if random_split_config is None:
            dataset_dict[dataset_id] = org_dataset
        else:
            split_dataset(org_dataset, random_split_config, dataset_id, dataset_dict)
        logger.info('dataset_id `{}`: {} sec'.format(dataset_id, time.time() - st))
    return dataset_dict


def get_all_datasets(datasets_config):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_dataset_dict(datasets_config[dataset_name])
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict


def build_data_loader(dataset, data_loader_config, distributed, accelerator=None):
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset = get_dataset_wrapper(dataset_wrapper_config['key'], dataset, **dataset_wrapper_config['kwargs'])
    elif cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)

    sampler_config = data_loader_config.get('sampler', None)
    sampler_kwargs = sampler_config.get('kwargs', None)
    if sampler_kwargs is None:
        sampler_kwargs = dict()

    if distributed and accelerator is None:
        sampler = DistributedSampler(dataset, **sampler_kwargs)
    else:
        sampler_cls_or_func = sampler_config['class_or_func']
        sampler = sampler_cls_or_func(dataset, **sampler_kwargs)

    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler = None if batch_sampler_config is None \
        else get_batch_sampler(batch_sampler_config['key'], sampler, **batch_sampler_config['kwargs'])
    collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    data_loader_kwargs = data_loader_config['kwargs']
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **data_loader_kwargs)
    return DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **data_loader_kwargs)


def build_data_loaders(dataset_dict, data_loader_configs, distributed, accelerator=None):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed, accelerator)
        data_loader_list.append(data_loader)
    return data_loader_list

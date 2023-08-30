import copy

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from ..common.constant import def_logger
from ..datasets.registry import get_collate_func, get_batch_sampler, get_dataset_wrapper
from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def split_dataset(dataset, lengths=None, generator_seed=None, sub_splits_configs=None, dataset_id=None):
    org_dataset_length = len(dataset)
    if dataset_id is not None:
        logger.info('Splitting `{}` dataset ({} samples in total)'.format(dataset_id, org_dataset_length))
    if lengths is None:
        lengths = (9, 1)

    total_length = sum(lengths)
    if total_length != org_dataset_length:
        lengths = [int((l / total_length) * org_dataset_length) for l in lengths]
        if len(lengths) > 1 and sum(lengths) != org_dataset_length:
            lengths[-1] = org_dataset_length - sum(lengths[:-1])

    sub_datasets = random_split(dataset, lengths) if generator_seed is None \
        else random_split(dataset, lengths, generator=torch.Generator().manual_seed(generator_seed))
    if sub_splits_configs is None:
        return sub_datasets

    # Deep-copy dataset to configure transforms independently as dataset in Subset class is shallow-copied
    for sub_dataset in sub_datasets:
        sub_dataset.dataset = copy.deepcopy(sub_dataset.dataset)

    assert len(sub_datasets) == len(sub_splits_configs), \
        'len(lengths) `{}` should be equal to len(sub_splits_configs) `{}`'.format(len(sub_datasets),
                                                                                   len(sub_splits_configs))
    for sub_dataset, sub_split_kwargs in zip(sub_datasets, sub_splits_configs):
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
    return sub_datasets


def build_data_loader(dataset, data_loader_config, distributed, accelerator=None):
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset_wrapper_args = dataset_wrapper_config.get('args', None)
        dataset_wrapper_kwargs = dataset_wrapper_config.get('kwargs', None)
        if dataset_wrapper_args is None:
            dataset_wrapper_args = list()
        if dataset_wrapper_kwargs is None:
            dataset_wrapper_kwargs = dict()
        dataset_wrapper_cls_or_func = get_dataset_wrapper(dataset_wrapper_config['key'])
        dataset = dataset_wrapper_cls_or_func(dataset, *dataset_wrapper_args, **dataset_wrapper_kwargs)
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
    batch_sampler_cls_or_func = None if batch_sampler_config is None else get_batch_sampler(batch_sampler_config['key'])
    batch_sampler = None if batch_sampler_cls_or_func is None \
        else batch_sampler_cls_or_func(sampler, **batch_sampler_config['kwargs'])
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

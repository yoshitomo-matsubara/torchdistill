import copy
import time

import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import PhotoTour, Kinetics400, HMDB51, UCF101, Cityscapes, CocoCaptions, CocoDetection, \
    SBDataset, VOCSegmentation, VOCDetection

from ..common.constant import def_logger
from ..datasets.coco import ImageToTensor, Compose, CocoRandomHorizontalFlip, get_coco
from ..datasets.registry import DATASET_DICT, TRANSFORM_DICT, \
    get_collate_func, get_sample_loader, get_batch_sampler, get_dataset_wrapper
from ..datasets.transform import CustomCompose
from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)

TRANSFORM_DICT.update(torchvision.transforms.__dict__)


def load_coco_dataset(img_dir_path, ann_file_path, annotated_only, random_horizontal_flip=None, is_segment=False,
                      transforms=None, jpeg_quality=None):
    if transforms is None:
        transform_list = [ImageToTensor(jpeg_quality)]
        if random_horizontal_flip is not None and not is_segment:
            transform_list.append(CocoRandomHorizontalFlip(random_horizontal_flip))
        transforms = Compose(transform_list)
    return get_coco(img_dir_path=img_dir_path, ann_file_path=ann_file_path,
                    transforms=transforms, annotated_only=annotated_only, is_segment=is_segment)


def build_transform(transform_configs, compose_cls=None):
    if not isinstance(transform_configs, (dict, list)) or len(transform_configs) == 0:
        return None

    if isinstance(compose_cls, str):
        compose_cls = TRANSFORM_DICT[compose_cls]

    component_list = list()
    for component_config in transform_configs:
        kwargs = component_config.get('kwargs', dict())
        if kwargs is None:
            kwargs = dict()

        component = TRANSFORM_DICT[component_config['type']](**kwargs)
        component_list.append(component)
    return torchvision.transforms.Compose(component_list) if compose_cls is None else compose_cls(component_list)


def get_torchvision_dataset(dataset_cls, dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    transform_compose_cls_name = dataset_kwargs.pop('transform_compose_cls', None)
    transform = build_transform(dataset_kwargs.pop('transform_configs', None), compose_cls=transform_compose_cls_name)
    target_transform_compose_cls_name = dataset_kwargs.pop('target_transform_compose_cls', None)
    target_transform = build_transform(dataset_kwargs.pop('target_transform_configs', None),
                                       compose_cls=target_transform_compose_cls_name)
    transforms_compose_cls_name = dataset_kwargs.pop('transforms_compose_cls', None)
    transforms = \
        build_transform(dataset_kwargs.pop('transforms_configs', None), compose_cls=transforms_compose_cls_name)
    if 'loader' in dataset_kwargs:
        loader_config = dataset_kwargs.pop('loader')
        loader_type = loader_config['type']
        loader_kwargs = loader_config.get('kwargs', None)
        loader = get_sample_loader(loader_type) if loader_kwargs is None \
            else get_sample_loader(loader_type, **loader_kwargs)
        dataset_kwargs['loader'] = loader

    # For datasets without target_transform
    if dataset_cls in (PhotoTour, Kinetics400, HMDB51, UCF101):
        return dataset_cls(transform=transform, **dataset_kwargs)
    # For datasets with transforms
    if dataset_cls in (Cityscapes, CocoCaptions, CocoDetection, SBDataset, VOCSegmentation, VOCDetection):
        return dataset_cls(transform=transform, target_transform=target_transform,
                           transforms=transforms, **dataset_kwargs)
    return dataset_cls(transform=transform, target_transform=target_transform, **dataset_kwargs)


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
        transform = build_transform(sub_split_kwargs.pop('transform_configs', None))
        target_transform = build_transform(sub_split_kwargs.pop('transform_configs', None))
        transforms = build_transform(sub_split_kwargs.pop('transforms_configs', None))
        if hasattr(sub_dataset.dataset, 'transform') and transform is not None:
            sub_dataset.dataset.transform = transform
        if hasattr(sub_dataset.dataset, 'target_transform') and target_transform is not None:
            sub_dataset.dataset.target_transform = target_transform
        if hasattr(sub_dataset.dataset, 'transforms') and transforms is not None:
            sub_dataset.dataset.transforms = transforms
        dataset_dict[sub_dataset_id] = sub_dataset


def get_dataset_dict(dataset_config):
    dataset_type = dataset_config['type']
    dataset_dict = dict()
    if dataset_type == 'cocodetect':
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            split_config = dataset_splits_config[split_name]
            is_segment = split_config.get('is_segment', False)
            compose_cls = CustomCompose if is_segment else None
            transforms = build_transform(split_config.get('transforms_configs', None), compose_cls=compose_cls)
            dataset_dict[split_config['dataset_id']] =\
                load_coco_dataset(split_config['images'], split_config['annotations'],
                                  split_config['annotated_only'], split_config.get('random_horizontal_flip', None),
                                  is_segment, transforms, split_config.get('jpeg_quality', None))
    elif dataset_type in DATASET_DICT:
        dataset_cls_or_func = DATASET_DICT[dataset_type]
        is_torchvision = dataset_type in torchvision.datasets.__dict__
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            st = time.time()
            logger.info('Loading {} data'.format(split_name))
            split_config = dataset_splits_config[split_name]
            org_dataset = get_torchvision_dataset(dataset_cls_or_func, split_config['kwargs']) if is_torchvision \
                else dataset_cls_or_func(**split_config['kwargs'])
            dataset_id = split_config['dataset_id']
            random_split_config = split_config.get('random_split', None)
            if random_split_config is None:
                dataset_dict[dataset_id] = org_dataset
            else:
                split_dataset(org_dataset, random_split_config, dataset_id, dataset_dict)
            logger.info('dataset_id `{}`: {} sec'.format(dataset_id, time.time() - st))
    else:
        raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))
    return dataset_dict


def get_all_datasets(datasets_config):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_dataset_dict(datasets_config[dataset_name])
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict


def build_data_loader(dataset, data_loader_config, distributed, accelerator=None):
    num_workers = data_loader_config['num_workers']
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset = get_dataset_wrapper(dataset_wrapper_config['name'], dataset, **dataset_wrapper_config['kwargs'])
    elif cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)

    sampler = DistributedSampler(dataset) if distributed and accelerator is None \
        else RandomSampler(dataset) if data_loader_config.get('random_sample', False) else SequentialSampler(dataset)
    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler = None if batch_sampler_config is None \
        else get_batch_sampler(batch_sampler_config['type'], sampler, **batch_sampler_config['kwargs'])
    collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    drop_last = data_loader_config.get('drop_last', False)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                          collate_fn=collate_fn, drop_last=drop_last)

    batch_size = data_loader_config['batch_size']
    pin_memory = data_loader_config.get('pin_memory', True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last)


def build_data_loaders(dataset_dict, data_loader_configs, distributed, accelerator=None):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed, accelerator)
        data_loader_list.append(data_loader)
    return data_loader_list

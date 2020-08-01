import time

import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import PhotoTour, VOCDetection, Kinetics400, HMDB51, UCF101

from kdkit.common.constant import def_logger
from kdkit.datasets.coco import ImageToTensor, Compose, CocoRandomHorizontalFlip, get_coco, coco_collate_fn
from kdkit.datasets.sample_loader import get_sample_loader
from kdkit.datasets.sampler import get_batch_sampler
from kdkit.datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset, get_dataset_wrapper

logger = def_logger.getChild(__name__)

DATASET_DICT = torchvision.datasets.__dict__
TRANSFORMS_DICT = torchvision.transforms.__dict__


def load_coco_dataset(img_dir_path, ann_file_path, annotated_only, random_horizontal_flip=None):
    transform_list = [ImageToTensor()]
    if random_horizontal_flip is not None:
        transform_list.append(CocoRandomHorizontalFlip(random_horizontal_flip))
    return get_coco(img_dir_path=img_dir_path, ann_file_path=ann_file_path,
                    transforms=Compose(transform_list), annotated_only=annotated_only)


def build_transform(transform_params_config):
    if not isinstance(transform_params_config, dict) or len(transform_params_config) == 0:
        return None

    component_list = list()
    for component_key in sorted(transform_params_config.keys()):
        component_config = transform_params_config[component_key]
        params_config = component_config.get('params', dict())
        if params_config is None:
            params_config = dict()

        component = TRANSFORMS_DICT[component_config['type']](**params_config)
        component_list.append(component)
    return transforms.Compose(component_list)


def get_official_dataset(dataset_cls, dataset_params_config):
    params_config = dataset_params_config.copy()
    transform = build_transform(params_config.pop('transform_params', None))
    target_transform = build_transform(params_config.pop('transform_params', None))
    if 'loader' in params_config:
        loader_config = params_config.pop('loader')
        loader_type = loader_config['type']
        loader_params_config = loader_config.get('params', None)
        loader = get_sample_loader(loader_type) if loader_params_config is None \
            else get_sample_loader(loader_type, **loader_params_config)
        params_config['loader'] = loader

    # For datasets without target_transform
    if dataset_cls in (PhotoTour, VOCDetection, Kinetics400, HMDB51, UCF101):
        return dataset_cls(transform=transform, **params_config)
    return dataset_cls(transform=transform, target_transform=target_transform, **params_config)


def get_dataset_dict(dataset_config):
    dataset_type = dataset_config['type']
    dataset_dict = dict()
    if dataset_type == 'cocodetect':
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            split_config = dataset_splits_config[split_name]
            dataset_dict[split_config['dataset_id']] =\
                load_coco_dataset(split_config['images'], split_config['annotations'],
                                  split_config['annotated_only'], split_config.get('random_horizontal_flip', None))
    elif dataset_type in DATASET_DICT:
        dataset_cls = DATASET_DICT[dataset_type]
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            st = time.time()
            logger.info('Loading {} data'.format(split_name))
            split_config = dataset_splits_config[split_name]
            dataset_dict[split_config['dataset_id']] = get_official_dataset(dataset_cls, split_config['params'])
            logger.info('{} sec'.format(time.time() - st))
    else:
        raise ValueError('dataset_type `{}` is not expected'.format(dataset_type))
    return dataset_dict


def get_all_dataset(datasets_config):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_dataset_dict(datasets_config[dataset_name])
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict


def build_data_loader(dataset, data_loader_config, distributed):
    num_workers = data_loader_config['num_workers']
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset = get_dataset_wrapper(dataset_wrapper_config['name'], dataset, **dataset_wrapper_config['params'])
    elif cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)

    sampler = DistributedSampler(dataset) if distributed \
        else RandomSampler(dataset) if data_loader_config.get('random_sample', False) else SequentialSampler(dataset)
    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler = None if batch_sampler_config is None \
        else get_batch_sampler(dataset, batch_sampler_config['type'], sampler, **batch_sampler_config['params'])
    collate_fn = coco_collate_fn if data_loader_config.get('collate_fn', None) == 'coco_collate_fn' else None
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)

    batch_size = data_loader_config['batch_size']
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


def build_data_loaders(dataset_dict, data_loader_configs, distributed):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed)
        data_loader_list.append(data_loader)
    return data_loader_list

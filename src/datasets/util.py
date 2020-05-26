import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from common.constant import def_logger
from datasets.coco import ImageToTensor, Compose, CocoRandomHorizontalFlip, get_coco, coco_collate_fn
from datasets.sampler import get_batch_sampler
from datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def load_image_folder_dataset(dir_path, data_aug, rough_size, input_size, normalizer, split_name):
    input_size = tuple(input_size)
    # Data loading
    st = time.time()
    if data_aug:
        logger.info('Loading {} data'.format(split_name))
        train_dataset = ImageFolder(
            dir_path,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
        logger.info('{} sec'.format(time.time() - st))
        return train_dataset

    logger.info('Loading {} data'.format(split_name))
    eval_dataset = ImageFolder(
        dir_path,
        transforms.Compose([
            transforms.Resize(rough_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalizer,
        ]))
    logger.info('{} sec'.format(time.time() - st))
    return eval_dataset


def load_coco_dataset(img_dir_path, ann_file_path, annotated_only, random_horizontal_flip=None):
    transform_list = [ImageToTensor()]
    if random_horizontal_flip is not None:
        transform_list.append(CocoRandomHorizontalFlip(random_horizontal_flip))
    return get_coco(img_dir_path=img_dir_path, ann_file_path=ann_file_path,
                    transforms=Compose(transform_list), annotated_only=annotated_only)


def get_dataset_dict(dataset_config):
    dataset_type = dataset_config['type']
    dataset_dict = dict()
    if dataset_type == 'imagefolder':
        rough_size = dataset_config['rough_size']
        input_size = dataset_config['input_size']
        normalizer = transforms.Normalize(**dataset_config['normalizer'])
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            split_config = dataset_splits_config[split_name]
            dataset_dict[split_config['dataset_id']] =\
                load_image_folder_dataset(split_config['images'], split_config['data_aug'], rough_size,
                                          input_size, normalizer, split_name)
    elif dataset_type == 'cocodetect':
        dataset_splits_config = dataset_config['splits']
        for split_name in dataset_splits_config.keys():
            split_config = dataset_splits_config[split_name]
            dataset_dict[split_config['dataset_id']] =\
                load_coco_dataset(split_config['images'], split_config['annotations'],
                                  split_config['annotated_only'], split_config.get('random_horizontal_flip', None))
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
    batch_size, num_workers = data_loader_config['batch_size'], data_loader_config['num_workers']
    cache_dir_path = data_loader_config.get('cache_output', None)
    if cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)

    sampler = DistributedSampler(dataset) if distributed \
        else RandomSampler(dataset) if data_loader_config.get('random_sample', False) else SequentialSampler(dataset)
    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler = None if batch_sampler_config is None \
        else get_batch_sampler(dataset, batch_sampler_config['name'], sampler, **batch_sampler_config['params'])
    collate_fn = coco_collate_fn if batch_sampler_config.get('collate_fn', None) == 'coco_collate_fn' else None
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
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

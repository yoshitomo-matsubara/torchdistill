import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from wrappers.dataset import default_idx2subpath, BaseDatasetWrapper, CacheableDataset


def load_image_folder_dataset(dir_path, data_aug, rough_size, input_size, normalizer, split_name):
    input_size = tuple(input_size)
    # Data loading code
    st = time.time()
    if data_aug:
        print('Loading {} data'.format(split_name))
        train_dataset = ImageFolder(
            dir_path,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
        print('\t', time.time() - st)
        return train_dataset

    print('Loading {} data'.format(split_name))
    eval_dataset = ImageFolder(
        dir_path,
        transforms.Compose([
            transforms.Resize(rough_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalizer,
        ]))
    print('\t', time.time() - st)
    return eval_dataset


def get_dataset_dict(dataset_config):
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
                load_image_folder_dataset(split_config['images'], split_config['data_aug'], rough_size,
                                          input_size, normalizer, split_name)
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
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


def build_data_loaders(dataset_dict, data_loader_configs, distributed):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed)
        data_loader_list.append(data_loader)
    return data_loader_list

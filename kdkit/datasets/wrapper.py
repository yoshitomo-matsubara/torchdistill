import os

import torch
from torch.utils.data import Dataset

from myutils.common import file_util


def default_idx2subpath(index):
    digits_str = '{:04d}'.format(index)
    return os.path.join(digits_str[-4:], digits_str)


class BaseDatasetWrapper(Dataset):
    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target = self.org_dataset.__getitem__(index)
        return sample, target, dict()

    def __len__(self):
        return len(self.org_dataset)


class CacheableDataset(BaseDatasetWrapper):
    def __init__(self, org_dataset, cache_dir_path, idx2subpath_func=None, ext='.pt'):
        super().__init__(org_dataset)
        self.cache_dir_path = cache_dir_path
        self.idx2subath_func = str if idx2subpath_func is None else idx2subpath_func
        self.ext = ext

    def __getitem__(self, index):
        sample, target, supp_dict = super().__getitem__(index)
        cache_file_path = os.path.join(self.cache_dir_path, self.idx2subath_func(index) + self.ext)
        if file_util.check_if_exists(cache_file_path):
            cached_data = torch.load(cache_file_path)
            supp_dict['cached_data'] = cached_data

        supp_dict['cache_file_path'] = cache_file_path
        return sample, target, supp_dict

    def __len__(self):
        return len(self.org_dataset)

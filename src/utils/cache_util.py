import os

import torch

from myutils.common import file_util


def cache_value_if_key_not_exist(key, value, output_dir_path, ext='.pt'):
    hash_id = str(abs(hash(key)))
    sub_dir_name = hash_id[-4:]
    output_file_path = os.path.join(output_dir_path, sub_dir_name, hash_id + ext)
    if not file_util.check_if_exists(output_file_path):
        file_util.make_parent_dirs(output_file_path)
        torch.save(value, output_file_path)


def load_cached_value_if_key_exist(key, dir_path, ext='.pt'):
    hash_id = str(abs(hash(key)))
    sub_dir_name = hash_id[-4:]
    file_path = os.path.join(dir_path, sub_dir_name, hash_id + ext)
    return torch.load(file_path) if file_util.check_if_exists(file_path) else None

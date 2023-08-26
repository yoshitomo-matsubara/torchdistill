import os
import pickle
import sys
from pathlib import Path


def check_if_exists(file_path):
    """
    Checks if a file/dir exists.

    :param file_path: file/dir path
    :type file_path: str
    :return: True if the given file exists
    :rtype: bool
    """
    return file_path is not None and os.path.exists(file_path)


def get_file_path_list(dir_path, is_recursive=False, is_sorted=False):
    """
    Gets file paths for a given dir path.

    :param dir_path: dir path
    :type dir_path: str
    :param is_recursive: if True, get file paths recursively
    :type is_recursive: bool
    :param is_sorted: if True, sort file paths in ascending order
    :type is_sorted: bool
    :return: list of file paths
    :rtype: list[str]
    """
    file_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isfile(path):
            file_list.append(path)
        elif is_recursive:
            file_list.extend(get_file_path_list(path, is_recursive))
    return sorted(file_list) if is_sorted else file_list


def get_dir_path_list(dir_path, is_recursive=False, is_sorted=False):
    """
    Gets dir paths for a given dir path.

    :param dir_path: dir path
    :type dir_path: str
    :param is_recursive: if True, get dir paths recursively
    :type is_recursive: bool
    :param is_sorted: if True, sort dir paths in ascending order
    :type is_sorted: bool
    :return: list of dir paths
    :rtype: list[str]
    """
    dir_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isdir(path):
            dir_list.append(path)
        elif is_recursive:
            dir_list.extend(get_dir_path_list(path, is_recursive))
    return sorted(dir_list) if is_sorted else dir_list


def make_dirs(dir_path):
    """
    Makes a directory and its parent directories.

    :param dir_path: dir path
    :type dir_path: str
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_parent_dirs(file_path):
    """
    Makes parent directories.

    :param file_path: file path
    :type file_path: str
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, file_path):
    """
    Saves a serialized object as a file.

    :param obj: object to be serialized
    :type obj: Any
    :param file_path: output file path
    :type file_path: str
    """
    make_parent_dirs(file_path)
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pickle(file_path):
    """
    Loads a deserialized object from a file.

    :param file_path: serialized file path
    :type file_path: str
    :return: deserialized object
    :rtype: Any
    """
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def get_binary_object_size(obj, unit_size=1024):
    """
    Computes the size of object in bytes after serialization.

    :param obj: object
    :type obj: Any
    :param unit_size: unit file size
    :type unit_size: int or float
    :return: size of object in bytes, divided by the `unit_size`
    :rtype: float
    """
    return sys.getsizeof(pickle.dumps(obj)) / unit_size

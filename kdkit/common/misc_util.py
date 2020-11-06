import inspect
import os
import sys


def check_if_plottable():
    return os.environ.get('DISPLAY', '') != ''


def get_classes(package_name, require_names=False):
    members = inspect.getmembers(sys.modules[package_name], inspect.isclass)
    if require_names:
        return members
    return [obj for _, obj in members]


def get_classes_as_dict(package_name, is_lower=True):
    members = get_classes(package_name, require_names=True)
    class_dict = dict()
    for name, obj in members:
        class_dict[name.lower() if is_lower else name] = obj
    return class_dict


def get_functions(package_name, require_names=False):
    members = inspect.getmembers(sys.modules[package_name], inspect.isfunction)
    if require_names:
        return members
    return [obj for _, obj in members]


def get_functions_as_dict(package_name, is_lower=True):
    members = get_functions(package_name, require_names=True)
    func_dict = dict()
    for name, obj in members:
        func_dict[name.lower() if is_lower else name] = obj
    return func_dict

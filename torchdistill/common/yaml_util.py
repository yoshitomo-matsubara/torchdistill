import os

import yaml

from .constant import def_logger
from .main_util import import_get, import_call

logger = def_logger.getChild(__name__)


def yaml_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def yaml_pathjoin(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.expanduser(os.path.join(*[str(i) for i in seq]))


def yaml_import_get(loader, node):
    entry = loader.construct_mapping(node, deep=True)
    return import_get(**entry)


def yaml_import_call(loader, node):
    entry = loader.construct_mapping(node, deep=True)
    return import_call(**entry)


def yaml_getattr(loader, node):
    args = loader.construct_sequence(node, deep=True)
    return getattr(*args)


def load_yaml_file(yaml_file_path, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)
        yaml.add_constructor('!import_get', yaml_import_get, Loader=yaml.FullLoader)
        yaml.add_constructor('!import_call', yaml_import_call, Loader=yaml.FullLoader)
        yaml.add_constructor('!getattr', yaml_getattr, Loader=yaml.FullLoader)
    with open(yaml_file_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)

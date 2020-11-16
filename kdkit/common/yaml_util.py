import os
import yaml


def yaml_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def yaml_pathjoin(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.expanduser(os.path.join(*[str(i) for i in seq]))


def load_yaml_file(yaml_file_path, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)
    with open(yaml_file_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)

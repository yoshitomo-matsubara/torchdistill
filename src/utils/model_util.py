from collections import OrderedDict

from torch.nn import Sequential, DataParallel
from torch.nn.parallel import DistributedDataParallel

from myutils.pytorch.module_util import get_module, freeze_module_params


def redesign_model(org_model, model_config, model_label, device_ids=None):
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        print('Using original {} model ...'.format(model_label))
        return org_model

    print('Redesigning {} model ...'.format(model_label))
    module_dict = OrderedDict()
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    for module_path in module_paths:
        # TODO: load adaptation modules defined as lists of list (name, module) if module_path matches;
        #  Names of adaptation modules should not conflict with those of original modules
        module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)

        module_dict[module_path.replace('.', '__attr__')] = module

    model = Sequential(module_dict)
    wrapper = model_config.get('wrapper', None)
    if wrapper is not None:
        if wrapper == 'DataParallel':
            model = DataParallel(model, device_ids=device_ids)
        elif wrapper == 'DistributedDataParallel':
            model = DistributedDataParallel(model, device_ids=device_ids)

    if not model_config.get('requires_grad', True):
        freeze_module_params(model)
    return model

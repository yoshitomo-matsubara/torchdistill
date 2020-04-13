from collections import OrderedDict

from torch.nn import Sequential, DataParallel
from torch.nn.parallel import DistributedDataParallel

from models.adaptation import get_adaptation_module
from myutils.pytorch.module_util import get_module, freeze_module_params


def wrap_model(model, model_config, device, device_ids=None, distributed=False):
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    model.to(device)
    if wrapper is not None and device.type.startswith('cuda'):
        if wrapper == 'DistributedDataParallel' and distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif wrapper in {'DataParallel', 'DistributedDataParallel'}:
            model = DataParallel(model, device_ids=device_ids)
    return model


def redesign_model(org_model, model_config, model_label):
    print('[{} model]'.format(model_label))
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        print('\tUsing the original {} model'.format(model_label))
        if len(frozen_module_path_set) > 0:
            print('\tFrozen module(s): {}'.format(frozen_module_path_set))

        for frozen_module_path in frozen_module_path_set:
            module = get_module(org_model, frozen_module_path)
            freeze_module_params(module)
        return org_model

    print('\tRedesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        print('\tFrozen module(s): {}'.format(frozen_module_path_set))

    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())
    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['type'], **adaptation_config['params'])
        else:
            module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)
        module_dict[module_path.replace('.', '__attr__')] = module

    model = Sequential(module_dict)
    return model

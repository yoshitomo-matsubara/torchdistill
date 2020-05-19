from collections import OrderedDict

from torch.nn import Sequential

from common.constant import def_logger
from models.adaptation import get_adaptation_module
from myutils.pytorch.module_util import get_module, freeze_module_params

logger = def_logger.getChild(__name__)


def redesign_model(org_model, model_config, model_label):
    logger.info('[{} model]'.format(model_label))
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the original {} model'.format(model_label))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

        for frozen_module_path in frozen_module_path_set:
            module = get_module(org_model, frozen_module_path)
            freeze_module_params(module)
        return org_model

    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

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

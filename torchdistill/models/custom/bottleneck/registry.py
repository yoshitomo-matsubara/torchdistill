from ....common.constant import def_logger

BOTTLENECK_PROCESSOR_DICT = dict()

logger = def_logger.getChild(__name__)


def register_bottleneck_processor(cls):
    BOTTLENECK_PROCESSOR_DICT[cls.__name__] = cls
    return cls


def get_bottleneck_processor(class_name, *args, **kwargs):
    if class_name not in BOTTLENECK_PROCESSOR_DICT:
        logger.info('No bottleneck processor called `{}` is registered.'.format(class_name))
        return None

    instance = BOTTLENECK_PROCESSOR_DICT[class_name](*args, **kwargs)
    return instance

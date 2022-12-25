from ....common.constant import def_logger

BOTTLENECK_PROCESSOR_DICT = dict()

logger = def_logger.getChild(__name__)


def register_bottleneck_processor(arg=None, **kwargs):
    def _register_bottleneck_processor(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        BOTTLENECK_PROCESSOR_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_bottleneck_processor(arg)
    return _register_bottleneck_processor


def get_bottleneck_processor(key, *args, **kwargs):
    if key not in BOTTLENECK_PROCESSOR_DICT:
        logger.info('No bottleneck processor called `{}` is registered.'.format(key))
        return None

    instance = BOTTLENECK_PROCESSOR_DICT[key](*args, **kwargs)
    return instance

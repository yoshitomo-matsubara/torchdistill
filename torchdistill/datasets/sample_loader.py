from io import BytesIO

from PIL import Image

from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

SAMPLE_LOADER_CLASS_DICT = dict()
SAMPLE_LOADER_FUNC_DICT = dict()


def register_sample_loader_class(arg=None, **kwargs):
    def _register_sample_loader_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SAMPLE_LOADER_CLASS_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_sample_loader_class(arg)
    return _register_sample_loader_class


def register_sample_loader_func(arg=None, **kwargs):
    def _register_sample_loader_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        SAMPLE_LOADER_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_sample_loader_func(arg)
    return _register_sample_loader_func


@register_sample_loader_class
class JpegCompressionLoader(object):
    def __init__(self, jpeg_quality=None):
        self.jpeg_quality = jpeg_quality
        logger.info('{} uses jpeg quality = `{}`'.format(self.__class__.__name__, jpeg_quality))

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.jpeg_quality is not None:
                img_buffer = BytesIO()
                img.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
                img = Image.open(img_buffer)
            return img


def get_sample_loader(obj_name, *args, **kwargs):
    if obj_name is None:
        return None
    elif obj_name in SAMPLE_LOADER_CLASS_DICT:
        return SAMPLE_LOADER_CLASS_DICT[obj_name](*args, **kwargs)
    elif obj_name in SAMPLE_LOADER_FUNC_DICT:
        return SAMPLE_LOADER_FUNC_DICT[obj_name]
    raise ValueError('No sample loader `{}` registered.'.format(obj_name))

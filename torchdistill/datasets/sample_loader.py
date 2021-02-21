from io import BytesIO

from PIL import Image

from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

SAMPLE_LOADER_CLASS_DICT = dict()
SAMPLE_LOADER_FUNC_DICT = dict()


def register_sample_loader_class(cls):
    SAMPLE_LOADER_CLASS_DICT[cls.__name__] = cls
    return cls


def register_sample_loader_func(func):
    SAMPLE_LOADER_FUNC_DICT[func.__name__] = func
    return func


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
    if obj_name not in SAMPLE_LOADER_CLASS_DICT and obj_name not in SAMPLE_LOADER_FUNC_DICT:
        logger.info('No sample loader called `{}` is registered.'.format(obj_name))
        return None

    if obj_name in SAMPLE_LOADER_CLASS_DICT:
        return SAMPLE_LOADER_CLASS_DICT[obj_name](*args, **kwargs)
    return SAMPLE_LOADER_FUNC_DICT[obj_name]

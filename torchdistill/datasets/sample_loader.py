from io import BytesIO

from PIL import Image

from .registry import register_sample_loader
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_sample_loader
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

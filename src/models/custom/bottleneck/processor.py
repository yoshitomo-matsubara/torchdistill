import os
from collections import namedtuple
from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import functional

from misc.log import def_logger
from myutils.common import file_util
from myutils.pytorch import tensor_util

logger = def_logger.getChild(__name__)
JpegCompressedTensor = namedtuple('JpegCompressedTensor', ['tensor_buffer', 'scale', 'zero_point'])
CLASS_DICT = dict()


def register_bottleneck_processor(cls):
    CLASS_DICT[cls.__name__] = cls
    return cls


@register_bottleneck_processor
class JpegCompressor(object):
    def __init__(self, jpeg_quality=95, tmp_dir_path=None):
        self.jpeg_quality = jpeg_quality
        self.tmp_dir_path = tmp_dir_path
        if tmp_dir_path is not None:
            file_util.make_dirs(tmp_dir_path)

    def compress(self, z, output_file_path):
        qz = tensor_util.quantize_tensor(z)
        img = Image.fromarray(qz.tensor.permute(1, 2, 0).cpu().numpy())
        buffer = BytesIO()
        img.save(buffer, format='jpeg', quality=self.jpeg_quality)
        if output_file_path is not None:
            img.save(output_file_path, format='jpeg', quality=self.jpeg_quality)
        return JpegCompressedTensor(tensor_buffer=buffer, scale=qz.scale, zero_point=qz.zero_point)

    def __call__(self, z):
        jc_tensor_list = list()
        for i, sub_z in enumerate(z):
            file_path = None
            if self.tmp_dir_path is not None:
                file_path = os.path.join(self.tmp_dir_path, '{}.jpg'.format(hash(sub_z)))
            jc_tensor_list.append(self.compress(sub_z, file_path))
        return jc_tensor_list


@register_bottleneck_processor
class JpegDecompressor(object):
    def __init__(self, prints_file_size=False):
        self.prints_file_size = prints_file_size

    def decompress(self, jc_tensor):
        if self.prints_file_size:
            logger.info('{:.4f} [KB]'.format(file_util.get_binary_object_size(jc_tensor)))
        img = Image.open(jc_tensor.tensor_buffer).convert('RGB')
        return jc_tensor.scale * (functional.to_tensor(img) * 255.0 - jc_tensor.zero_point)

    def __call__(self, jc_tensors):
        return torch.stack([self.decompress(jc_tensor) for jc_tensor in jc_tensors])


def get_bottleneck_processor(class_name, *args, **kwargs):
    if class_name not in CLASS_DICT:
        logger.info('No bottleneck processor called `{}` is registered.'.format(class_name))
        return None

    instance = CLASS_DICT[class_name](*args, **kwargs)
    return instance

import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

TRANSFORM_CLASS_DICT = dict()


def register_transform_class(cls):
    TRANSFORM_CLASS_DICT[cls.__name__] = cls
    return cls


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


@register_transform_class
class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


@register_transform_class
class CustomRandomResize(object):
    def __init__(self, min_size, max_size=None, jpeg_quality=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size

        self.max_size = max_size
        self.jpeg_quality = jpeg_quality

    def __call__(self, image, target):
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            image.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
            image = Image.open(img_buffer)

        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


@register_transform_class
class CustomRandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


@register_transform_class
class CustomRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


@register_transform_class
class CustomCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


@register_transform_class
class CustomToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


@register_transform_class
class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(obj_name, *args, **kwargs):
    if obj_name not in TRANSFORM_CLASS_DICT:
        logger.info('No transform called `{}` is registered.'.format(obj_name))
        return None
    return TRANSFORM_CLASS_DICT[obj_name](*args, **kwargs)

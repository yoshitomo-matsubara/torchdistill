from types import BuiltinFunctionType, BuiltinMethodType, FunctionType

import torch

COLLATE_FUNC_DICT = dict()


def register_collate_func(func):
    key = func.__name__ if isinstance(func, (BuiltinMethodType, BuiltinFunctionType, FunctionType)) \
        else type(func).__name__
    COLLATE_FUNC_DICT[key] = func
    return func


@register_collate_func
def coco_collate_fn(batch):
    return tuple(zip(*batch))


def cat_list(images, fill_value=0):
    if len(images) == 1 and not isinstance(images[0], torch.Tensor):
        return images

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


@register_collate_func
def coco_seg_collate_fn(batch):
    images, targets, supp_dicts = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets, supp_dicts


@register_collate_func
def coco_seg_eval_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def get_collate_func(func_name):
    if func_name not in COLLATE_FUNC_DICT:
        return None
    return COLLATE_FUNC_DICT[func_name]

import copy
import os
import random
from io import BytesIO

import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional

from torchdistill.datasets.registry import register_dataset


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class ImageToTensor(object):
    def __init__(self, jpeg_quality=None):
        self.jpeg_quality = jpeg_quality

    def __call__(self, image, target):
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            image.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
            image = Image.open(img_buffer)

        image = functional.to_tensor(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class CocoRandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)
            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target['keypoints'] = keypoints
        return image, target


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj['category_id'] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj['category_id'] = self.categories.index(obj['category_id'])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask4Detect(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj['segmentation'] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['masks'] = masks
        target['image_id'] = image_id
        if keypoints is not None:
            target['keypoints'] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] for obj in anno])
        target['area'] = area
        target['iscrowd'] = iscrowd
        return image, target


class ConvertCocoPolysToMask4Seg(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in anno)


def count_visible_keypoints(anno):
    return sum(sum(1 for v in ann['keypoints'][2::3] if v > 0) for ann in anno)


def has_valid_annotation(anno, min_keypoints_per_image=10):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if 'keypoints' not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


def remove_images_without_annotations(dataset, cat_list=None):
    assert isinstance(dataset, CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj['category_id'] in cat_list]
        if has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets['image_id'].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets['boxes']
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann['segmentation'] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CustomCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(os.path.expanduser(img_folder), os.path.expanduser(ann_file))
        self.additional_transforms = transforms

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        image_id = self.ids[index]
        target = dict(image_id=image_id, annotations=target)
        if self.additional_transforms is not None:
            img, target = self.additional_transforms(img, target)
        return img, target


def get_coco(img_dir_path, ann_file_path, transforms, annotated_only, is_segment):
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    t = [FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask4Seg()] if is_segment \
        else [ConvertCocoPolysToMask4Detect()]
    if transforms is not None:
        t.append(transforms)

    transforms = Compose(t)
    dataset = CocoDetection(img_dir_path, os.path.expanduser(ann_file_path), transforms=transforms) if is_segment \
        else CustomCocoDetection(img_dir_path, ann_file_path, transforms=transforms)
    if annotated_only:
        dataset = remove_images_without_annotations(dataset)
    return dataset


@register_dataset
def coco_dataset(img_dir_path, ann_file_path, annotated_only, random_horizontal_flip=None, is_segment=False,
                 transforms=None, jpeg_quality=None):
    if transforms is None:
        transform_list = [ImageToTensor(jpeg_quality)]
        if random_horizontal_flip is not None and not is_segment:
            transform_list.append(CocoRandomHorizontalFlip(random_horizontal_flip))
        transforms = Compose(transform_list)
    return get_coco(img_dir_path=img_dir_path, ann_file_path=ann_file_path,
                    transforms=transforms, annotated_only=annotated_only, is_segment=is_segment)

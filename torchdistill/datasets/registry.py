import torchvision

DATASET_DICT = dict()
DATASET_DICT.update(torchvision.datasets.__dict__)


def register_dataset(cls_or_func):
    DATASET_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func

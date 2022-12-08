import torchvision

DATASET_DICT = dict()
DATASET_DICT.update(torchvision.datasets.__dict__)


def register_dataset(*args, **kwargs):
    def _register_dataset(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_DICT[key] = cls_or_func
        return cls_or_func

    return _register_dataset

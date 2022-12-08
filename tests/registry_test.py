from unittest import TestCase

from torchdistill.datasets.registry import register_dataset, DATASET_DICT
from torchdistill.models.registry import get_model
from torchdistill.core.forward_proc import register_forward_proc_func, get_forward_proc_func, forward_batch_only


class RegistryTest(TestCase):
    def test_torch_hub(self):
        model_name = 'tf_mobilenetv3_large_100'
        repo_or_dir = 'rwightman/pytorch-image-models'
        kwargs = {'pretrained': True}
        mobilenet_v3 = get_model(model_name, repo_or_dir, **kwargs)
        assert type(mobilenet_v3).__name__ == 'MobileNetV3'

    def test_register_dataset(self):
        default_dataset_dict_size = len(DATASET_DICT)
        @register_dataset()
        class TestDataset1(object):
            def __init__(self):
                self.name = 'test1'

        assert 'TestDataset1' in DATASET_DICT
        assert len(DATASET_DICT) == default_dataset_dict_size + 1
        random_name = 'custom_test_dataset_name2'
        @register_dataset(key=random_name)
        class TestDataset2(object):
            def __init__(self):
                self.name = 'test2'

        assert len(DATASET_DICT) == default_dataset_dict_size + 2
        assert 'TestDataset1' in DATASET_DICT and random_name in DATASET_DICT and 'TestDataset2' not in DATASET_DICT

    def test_register_forward_proc_func(self):
        @register_forward_proc_func()
        def test_forward_proc1(model, batch):
            return model(batch)

        assert get_forward_proc_func('test_forward_proc1') == test_forward_proc1
        random_name = 'custom_forward_proc_name2'
        @register_forward_proc_func(key=random_name)
        def test_forward_proc2(model, batch, label):
            return model(batch, label)

        assert get_forward_proc_func(random_name) == test_forward_proc2 \
               and get_forward_proc_func('test_forward_proc2') == forward_batch_only

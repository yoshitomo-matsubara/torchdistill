from unittest import TestCase

from torchdistill.core.forward_proc import register_forward_proc_func, get_forward_proc_func, forward_batch_only
from torchdistill.datasets.collator import register_collate_func, get_collate_func
from torchdistill.datasets.registry import register_dataset, DATASET_DICT
from torchdistill.datasets.sample_loader import register_sample_loader_class, register_sample_loader_func, \
    get_sample_loader
from torchdistill.datasets.sampler import register_batch_sampler_class, BATCH_SAMPLER_CLASS_DICT
from torchdistill.datasets.transform import register_transform_class, get_transform
from torchdistill.datasets.wrapper import register_dataset_wrapper, get_dataset_wrapper
from torchdistill.models.registry import get_model
from torchdistill.losses.custom import register_custom_loss, CUSTOM_LOSS_CLASS_DICT


class RegistryTest(TestCase):
    def test_torch_hub(self):
        model_name = 'tf_mobilenetv3_large_100'
        repo_or_dir = 'rwightman/pytorch-image-models'
        kwargs = {'pretrained': True}
        mobilenet_v3 = get_model(model_name, repo_or_dir, **kwargs)
        assert type(mobilenet_v3).__name__ == 'MobileNetV3'

    def test_register_dataset(self):
        default_dataset_dict_size = len(DATASET_DICT)
        @register_dataset
        class TestDataset0(object):
            def __init__(self):
                self.name = 'test0'

        assert 'TestDataset0' in DATASET_DICT
        assert len(DATASET_DICT) == default_dataset_dict_size + 1

        @register_dataset()
        class TestDataset1(object):
            def __init__(self):
                self.name = 'test1'

        assert 'TestDataset1' in DATASET_DICT
        assert len(DATASET_DICT) == default_dataset_dict_size + 2
        random_name = 'custom_test_dataset_name2'

        @register_dataset(key=random_name)
        class TestDataset2(object):
            def __init__(self):
                self.name = 'test2'

        assert len(DATASET_DICT) == default_dataset_dict_size + 3
        assert 'TestDataset1' in DATASET_DICT and random_name in DATASET_DICT and 'TestDataset2' not in DATASET_DICT

    def test_register_forward_proc_func(self):
        @register_forward_proc_func
        def test_forward_proc0(model, batch):
            return model(batch)

        assert get_forward_proc_func('test_forward_proc0') == test_forward_proc0

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

    def test_register_collate_func(self):
        @register_collate_func
        def test_collate0(batch, label):
            return batch, label

        assert get_collate_func('test_collate0') == test_collate0

        @register_collate_func()
        def test_collate1(batch, label):
            return batch, label

        assert get_collate_func('test_collate1') == test_collate1
        random_name = 'custom_collate_name2'

        @register_collate_func(key=random_name)
        def test_collate2(batch, label):
            return batch, label

        assert get_collate_func(random_name) == test_collate2 \
               and get_collate_func('test_collate2') is None

    def test_register_sample_loader(self):
        @register_sample_loader_class
        class TestSampleLoader0(object):
            def __init__(self):
                self.name = 'test0'

        assert get_sample_loader('TestSampleLoader0') is not None

        @register_sample_loader_class()
        class TestSampleLoader1(object):
            def __init__(self):
                self.name = 'test1'

        assert get_sample_loader('TestSampleLoader1') is not None
        random_name = 'custom_sample_loader_class_name2'

        @register_sample_loader_class(key=random_name)
        class TestSampleLoader2(object):
            def __init__(self):
                self.name = 'test2'

        assert get_sample_loader(random_name) is not None

        @register_sample_loader_func
        def test_sample_loader0(batch):
            pass

        assert get_sample_loader('test_sample_loader0') == test_sample_loader0

        @register_sample_loader_func()
        def test_sample_loader1(batch, label):
            pass

        assert get_sample_loader('test_sample_loader1') == test_sample_loader1
        random_name = 'custom_sample_loader_func_name2'

        @register_sample_loader_func(key=random_name)
        def test_sample_loader2(batch, label):
            pass

        assert get_sample_loader(random_name) == test_sample_loader2 \
               and get_sample_loader('test_sample_loader2') is None

    def test_register_sampler(self):
        @register_batch_sampler_class
        class TestBatchSampler0(object):
            def __init__(self):
                self.name = 'test0'

        assert 'TestBatchSampler0' in BATCH_SAMPLER_CLASS_DICT

        @register_batch_sampler_class()
        class TestBatchSampler1(object):
            def __init__(self):
                self.name = 'test1'

        assert 'TestBatchSampler1' in BATCH_SAMPLER_CLASS_DICT
        random_name = 'custom_batch_sampler_class_name2'

        @register_batch_sampler_class(key=random_name)
        class TestBatchSampler2(object):
            def __init__(self):
                self.name = 'test2'

        assert random_name in BATCH_SAMPLER_CLASS_DICT

    def test_register_transform(self):
        @register_transform_class()
        class TestTransform0(object):
            def __init__(self):
                self.name = 'test0'

        assert get_transform('TestTransform0') is not None

        @register_transform_class()
        class TestTransform1(object):
            def __init__(self):
                self.name = 'test1'

        assert get_transform('TestTransform1') is not None
        random_name = 'custom_transform_class_name2'

        @register_transform_class(key=random_name)
        class TestTransform2(object):
            def __init__(self):
                self.name = 'test2'

        assert get_transform(random_name) is not None

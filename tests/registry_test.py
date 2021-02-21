from unittest import TestCase

from torchdistill.models.registry import get_model


class RegistryTest(TestCase):
    def test_torch_hub(self):
        model_name = 'tf_mobilenetv3_large_100'
        repo_or_dir = 'rwightman/pytorch-image-models'
        kwargs = {'pretrained': True}
        mobilenet_v3 = get_model(model_name, repo_or_dir, **kwargs)
        assert type(mobilenet_v3).__name__ == 'MobileNetV3'

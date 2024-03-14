from unittest import TestCase

from torchdistill.common.main_util import import_dependencies, import_get, import_call, import_call_method


class ImportUnitTest(TestCase):
    def test_import_dependencies(self):
        dependencies = [
            {'name': 'torchvision', 'package': 'models'},
            ['torch', 'nn'],
            'torch.nn.functional'
        ]

        flag = True
        try:
            import_dependencies(dependencies)
        except:
            flag = False
        assert flag

    def test_import_get(self):
        ver1 = import_get(key='__version__', package='torch')
        ver2 = import_get(key='torch.__version__')

        import torch
        true_ver = torch.__version__
        assert ver1 == true_ver
        assert ver2 == true_ver

    def test_import_call(self):
        kwargs1 = {'key': 'Softmax', 'package': 'torch.nn', 'init': {'kwargs': {'dim': 1}}}
        kwargs2 = {'key': 'Dropout', 'package': 'torch.nn'}

        softmax = import_call(**kwargs1)
        dropout = import_call(**kwargs2)

        from torch import nn
        assert str(softmax) == str(nn.Softmax(dim=1))
        assert str(dropout) == str(nn.Dropout())

    def test_import_call_method(self):
        kwargs1 = {'package': 'torchvision.models.alexnet', 'class_name': 'AlexNet_Weights', 'method_name': 'verify',
                   'init': {'kwargs': {'obj': 'AlexNet_Weights.IMAGENET1K_V1'}}}
        kwargs2 = {'package': 'torchvision.models.alexnet.AlexNet_Weights.verify',
                   'init': {'kwargs': {'obj': 'AlexNet_Weights.IMAGENET1K_V1'}}}

        weights1 = import_call_method(**kwargs1)
        weights2 = import_call_method(**kwargs2)
        from torchvision.models.alexnet import AlexNet_Weights
        true_weights = AlexNet_Weights.IMAGENET1K_V1
        assert weights1 == true_weights
        assert weights2 == true_weights

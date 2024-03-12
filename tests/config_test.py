from unittest import TestCase

from torchdistill.common.main_util import import_dependencies, import_get, import_call


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

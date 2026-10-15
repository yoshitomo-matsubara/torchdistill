from unittest import TestCase

from torch import nn

from torchdistill.common.module_util import get_module


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 3)

    def forward(self, x):
        return self.projection(x)


class WrappedModulePathTest(TestCase):
    def test_missing_path_matches_unwrapped_model(self):
        teacher = Teacher()
        wrapped = nn.DataParallel(teacher)
        for model in (teacher, wrapped):
            for path in ('projection', 'missing', 'missing.projection'):
                with self.subTest(wrapped=model is wrapped, path=path):
                    expected = teacher.projection if path == 'projection' else None
                    self.assertIs(get_module(model, path), expected)

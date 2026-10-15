from unittest import TestCase

import torch
from torch import nn

from torchdistill.core.forward_hook import ForwardHookManager, get_device_index


class MultiOutputTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 3)

    def forward(self, x):
        features = self.projection(x)
        return [features, {'labels': ['class'], 'scores': features.mean(dim=1)}]


class HookSequenceTest(TestCase):
    def test_nested_device_lookup(self):
        tensor = torch.ones(2)
        cases = [
            ([tensor], 'cpu'),
            (([{'features': tensor}],), 'cpu'),
            ({'metadata': 'class', 'features': tensor}, 'cpu'),
            (['metadata', {'features': tensor}], 'cpu'),
            ('metadata', None),
            (b'metadata', None),
            ((tensor,), 'cpu'),
            ({'features': tensor}, 'cpu'),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(get_device_index(value), expected)

    def test_pop_multi_output_teacher_from_cpu(self):
        for requires_input, requires_output in [(True, False), (False, True), (True, True)]:
            with self.subTest(input=requires_input, output=requires_output):
                model = MultiOutputTeacher()
                manager = ForwardHookManager('cpu')
                manager.add_hook_to_module(model, 'teacher', requires_input, requires_output)
                x = torch.randn(2, 2, requires_grad=True)
                try:
                    output = model(x)
                    captured = manager.pop_io_dict_from_device(torch.device('cpu'))['teacher']
                    if requires_input:
                        self.assertIs(captured['input'], x)
                    if requires_output:
                        self.assertIs(captured['output'], output)
                        captured['output'][0].sum().backward()
                        self.assertIsNotNone(x.grad)
                    self.assertTrue(all(not values for values in manager.io_dict['teacher'].values()))
                finally:
                    manager.clear()
                self.assertFalse(model._forward_hooks)

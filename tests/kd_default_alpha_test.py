from unittest import TestCase

import torch
from torch.distributions import Categorical, kl_divergence
from torch.nn.functional import cross_entropy

from torchdistill.losses.registry import get_mid_level_loss
import torchdistill.losses.mid_level  # noqa: registers losses


class KDDefaultAlphaTest(TestCase):
    def test_optional_alpha_matches_soft_only_loss_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for alpha_kwargs in ({}, {'alpha': None}, {'alpha': 0}):
                for loss_key in ('KDLoss', 'LogitStdKDLoss'):
                    with self.subTest(dtype=dtype, alpha_kwargs=alpha_kwargs, loss_key=loss_key):
                        self.check_loss(dtype, alpha_kwargs, loss_key)

    def test_explicit_weights_preserve_combined_loss_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for alpha_kwargs in ({'alpha': 0.3}, {'alpha': 0.3, 'beta': 0.2}, {'alpha': None, 'beta': 0.2}):
                for loss_key in ('KDLoss', 'LogitStdKDLoss'):
                    with self.subTest(dtype=dtype, alpha_kwargs=alpha_kwargs, loss_key=loss_key):
                        self.check_loss(dtype, alpha_kwargs, loss_key)

    def check_loss(self, dtype, alpha_kwargs, loss_key):
        temperature = 2.5
        student = torch.tensor([[1.0, -0.5, 0.2], [-0.2, 0.4, 1.3]], dtype=dtype, requires_grad=True)
        teacher = torch.tensor([[-0.3, 0.8, 0.1], [1.1, -0.6, 0.3]], dtype=dtype)
        targets = torch.tensor([1, 0])
        criterion = get_mid_level_loss(
            {'key': loss_key, 'kwargs': dict(
                student_module_path='logits', student_module_io='output',
                teacher_module_path='logits', teacher_module_io='output',
                temperature=temperature, **alpha_kwargs
            )}
        )
        actual = criterion({'logits': {'output': student}}, {'logits': {'output': teacher}}, targets)
        reference_student = student.detach().clone().requires_grad_()
        soft_student, soft_teacher = reference_student, teacher
        if loss_key == 'LogitStdKDLoss':
            soft_student = self.standardize(soft_student)
            soft_teacher = self.standardize(soft_teacher)
        expected = kl_divergence(
            Categorical(logits=soft_teacher / temperature), Categorical(logits=soft_student / temperature)
        ).mean()
        alpha = alpha_kwargs.get('alpha')
        if alpha is not None and alpha != 0:
            beta = alpha_kwargs.get('beta', 1 - alpha)
            expected = alpha * cross_entropy(reference_student, targets) + beta * temperature ** 2 * expected
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(student.grad, reference_student.grad)

    @staticmethod
    def standardize(logits):
        centered = logits - logits.mean(dim=-1, keepdim=True)
        sample_std = (centered.square().sum(dim=-1, keepdim=True) / (logits.shape[-1] - 1)).sqrt()
        return centered / (sample_std + 1e-7)

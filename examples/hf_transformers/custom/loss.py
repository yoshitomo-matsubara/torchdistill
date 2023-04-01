import torch
from torch import nn
from torch.nn import functional

from torchdistill.losses.registry import register_single_loss
from torchdistill.losses.util import register_func2extract_model_output


@register_func2extract_model_output
def extract_transformers_loss(student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    model_loss_dict = dict()
    model_loss_dict['loss'] = student_outputs.loss
    return model_loss_dict


@register_single_loss
class KDLoss4Transformer(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha

    def compute_soft_loss(self, student_logits, teacher_logits):
        return super().forward(torch.log_softmax(student_logits / self.temperature, dim=1),
                               torch.softmax(teacher_logits / self.temperature, dim=1))

    def compute_hard_loss(self, logits, positions, ignored_index):
        return functional.cross_entropy(logits, positions, reduction=self.cel_reduction, ignore_index=ignored_index)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = self.compute_soft_loss(student_output.logits, teacher_output.logits)
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = student_output.loss
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss

import torch
from torch import nn
from torch.nn import functional
from torchdistill.losses.single import register_single_loss, register_org_loss
from torchdistill.losses.util import register_func2extract_org_output


@register_func2extract_org_output
def extract_transformers_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    org_loss_dict['loss'] = student_outputs.loss
    return org_loss_dict


@register_org_loss
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


@register_single_loss
class KDLoss4QA(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha
        self.cel_reduction = 'mean' if reduction == 'batchmean' else reduction

    def compute_soft_loss(self, student_logits, teacher_logits):
        return super().forward(torch.log_softmax(student_logits / self.temperature, dim=-1),
                               torch.softmax(teacher_logits / self.temperature, dim=-1))

    def compute_hard_loss(self, logits, positions, ignored_index):
        return functional.cross_entropy(logits, positions, reduction=self.cel_reduction, ignore_index=ignored_index)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        _, student_start_logits, student_end_logits = student_output
        _, teacher_start_logits, teacher_end_logits = teacher_output
        start_soft_loss = self.compute_soft_loss(student_start_logits, teacher_start_logits)
        end_soft_loss = self.compute_soft_loss(student_end_logits, teacher_end_logits)
        soft_loss = (start_soft_loss + end_soft_loss) / 2
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        start_positions = targets['start_positions']
        end_positions = targets['end_positions']
        ignored_index = student_start_logits.size(1)
        start_hard_loss = self.compute_hard_loss(student_start_logits, start_positions, ignored_index)
        end_hard_loss = self.compute_hard_loss(student_end_logits, end_positions, ignored_index)
        hard_loss = (start_hard_loss + end_hard_loss) / 2
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss

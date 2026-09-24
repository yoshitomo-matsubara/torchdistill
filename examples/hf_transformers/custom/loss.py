import torch.nn.functional as F
from torch import nn

from torchdistill.common.constant import SELF_MODULE_PATH
from torchdistill.losses.registry import register_mid_level_loss
from torchdistill.losses.util import register_func2extract_model_output


@register_func2extract_model_output
def extract_transformers_loss(student_outputs, targets, **kwargs):
    model_loss_dict = dict()
    model_loss_dict['loss'] = student_outputs.loss
    return model_loss_dict


class TokenLevelKDLoss(nn.Module):
    """
    Base class for token-level KL divergence between the student and teacher next-token distributions.

    Subclasses only decide *which* positions are scored, by implementing :meth:`get_scored_positions`.
    The shift convention is shared: logits at position *t* predict the token at *t + 1*, so a slice
    ``[start - 1: -1]`` of the logits lines up with the tokens at ``[start:]``.

    :param kl_type: ``'forward'`` for KL(p_teacher ∥ p_student), which is teacher-covering, or ``'reverse'``
        for KL(p_student ∥ p_teacher), which is mode-seeking.
    :type kl_type: str
    :param student_module_path: student module path in the I/O dict to extract the logits from.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the student module.
    :type student_module_io: str
    :param teacher_module_path: teacher module path in the I/O dict to extract the logits from.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the teacher module.
    :type teacher_module_io: str
    """
    def __init__(
            self, kl_type='forward', student_module_path=SELF_MODULE_PATH, student_module_io='output',
            teacher_module_path=SELF_MODULE_PATH, teacher_module_io='output', **kwargs
    ):
        super().__init__()
        if kl_type not in ('forward', 'reverse'):
            raise ValueError('kl_type `{}` should be either `forward` or `reverse`'.format(kl_type))

        self.kl_type = kl_type
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io

    @staticmethod
    def extract_logits(io_dict, module_path, module_io):
        module_outputs = io_dict[module_path][module_io]
        return module_outputs.logits if hasattr(module_outputs, 'logits') else module_outputs

    def get_scored_positions(self, targets):
        """
        Returns the offset of the first scored token and the boolean mask over the scored positions.

        :param targets: sample batch, which the training script passes as the targets.
        :type targets: dict
        :return: index of the first scored token, and a mask of shape (batch size, sequence length - offset).
        :rtype: (int, torch.Tensor)
        """
        raise NotImplementedError

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_logits = self.extract_logits(student_io_dict, self.student_module_path, self.student_module_io)
        teacher_logits = self.extract_logits(teacher_io_dict, self.teacher_module_path, self.teacher_module_io)
        start_index, scored_mask = self.get_scored_positions(targets)
        if not scored_mask.any():
            # Differentiable zero, in case a batch has nothing to score
            return student_logits.sum() * 0.0

        # logits[:, start_index - 1: -1] predicts the tokens at [:, start_index:]
        student_log_probs = F.log_softmax(student_logits[:, start_index - 1: -1, :], dim=-1)[scored_mask]
        teacher_log_probs = F.log_softmax(teacher_logits[:, start_index - 1: -1, :], dim=-1)[scored_mask]
        if self.kl_type == 'forward':
            # KL(p_teacher ∥ p_student): the gradient flows through the student log-probabilities
            return F.kl_div(student_log_probs, teacher_log_probs.detach().exp(), reduction='batchmean')
        # KL(p_student ∥ p_teacher): the gradient flows through the student probabilities
        return F.kl_div(teacher_log_probs.detach(), student_log_probs.exp(), reduction='batchmean')


@register_mid_level_loss
class OnPolicyKDLoss(TokenLevelKDLoss):
    """
    Token-level KL divergence between the student and teacher distributions over on-policy sequences.

    The student generates the sequences (see ``generate_on_policy_sequences``), and both models are scored on
    them, so the divergence is measured where the student actually puts its probability mass. This follows the
    GKD framework, Gu et al., "Generalized Knowledge Distillation for Auto-regressive Sequence Models"
    (https://arxiv.org/abs/2306.13649).

    Only the generated (completion) tokens are scored. The prompt/completion boundary and the padding mask are
    read from the batch, which the training script passes as ``targets``.
    """
    def get_scored_positions(self, targets):
        return targets['prompt_length'], targets['completion_mask']


@register_mid_level_loss
class OffPolicyKDLoss(TokenLevelKDLoss):
    """
    Token-level KL divergence between the student and teacher distributions over the ground-truth sequences.

    Unlike :class:`OnPolicyKDLoss`, no generation is involved: both models are scored on the sequences in the
    dataset, which makes this usable for corpus pre-training as well as for instruction tuning. The scored
    positions are the ones the task loss covers, i.e. those whose label is not the ``-100`` ignore index.
    """
    def get_scored_positions(self, targets):
        # Labels are aligned with the input tokens, so position 0 is never predicted by any logit
        return 1, targets['labels'][:, 1:] != -100

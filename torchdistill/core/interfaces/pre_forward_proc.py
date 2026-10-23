import torch

from .registry import register_pre_forward_proc_func
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_pre_forward_proc_func
def default_pre_forward_process(self, *args, **kwargs):
    pass


@register_pre_forward_proc_func
def generate_on_policy_sequences(self, sample_batch=None, **generation_kwargs):
    """
    Collects on-policy data by letting the student generate completions for the prompts in ``sample_batch``.

    The generated sequences are written back into ``sample_batch`` so that the teacher and student forward
    processes score the same sequences, and so that the mid-level loss can restrict the KL divergence to the
    generated (completion) tokens:

    * ``on_policy_input_ids`` / ``on_policy_attention_mask``: prompt + generated completion.
    * ``completion_mask``: True at the generated positions up to and including the first EOS token.
    * ``prompt_length``: width of the left-padded prompt tensor, i.e. the offset at which completions start.

    Prompts are left-padded by the collate function, so every completion starts at the same offset and the
    boundary is a single integer rather than a per-sample index.

    :param self: training/distillation box.
    :type self: torchdistill.core.training.TrainingBox or torchdistill.core.distillation.DistillationBox
    :param sample_batch: sample batch, updated in place.
    :type sample_batch: dict
    :param generation_kwargs: kwargs for ``generate`` e.g., ``max_new_tokens``, ``do_sample``, ``temperature``.
    :type generation_kwargs: dict
    """
    if sample_batch is None:
        return

    if 'prompt_input_ids' not in sample_batch:
        raise KeyError(
            'on-policy rollouts require prompts (`prompt_input_ids` and `prompt_attention_mask`) in the batch'
        )

    # TrainingBox exposes the model as `model`, and DistillationBox as `student_model`
    student_model = self.student_model if hasattr(self, 'student_model') else self.model
    # Unwrap so that `generate` is reachable regardless of DataParallel/DDP/FSDP wrapping
    unwrapped_student_model = self.accelerator.unwrap_model(student_model) if self.accelerator is not None \
        else student_model
    pad_token_id = unwrapped_student_model.config.pad_token_id
    prompt_input_ids = sample_batch['prompt_input_ids']
    was_training = unwrapped_student_model.training
    unwrapped_student_model.eval()
    with torch.no_grad():
        generated_ids = unwrapped_student_model.generate(
            input_ids=prompt_input_ids, attention_mask=sample_batch['prompt_attention_mask'],
            pad_token_id=pad_token_id, **generation_kwargs
        )
    unwrapped_student_model.train(was_training)

    prompt_length = prompt_input_ids.shape[1]
    # Many causal LMs reuse EOS as the pad token (SmolLM2's chat template even puts it inside every prompt), so
    # comparing against pad_token_id would mask real tokens. The prompt part keeps the collator's mask, and a
    # completion is valid up to and including its first EOS; `generate` only pads after that
    generation_config = generation_kwargs.get('generation_config', unwrapped_student_model.generation_config)
    eos_token_id = generation_kwargs.get('eos_token_id', generation_config.eos_token_id)
    completion_ids = generated_ids[:, prompt_length:]
    if eos_token_id is None:
        completion_mask = torch.ones_like(completion_ids, dtype=torch.bool)
    else:
        is_eos = torch.isin(completion_ids, torch.tensor(eos_token_id, device=completion_ids.device).view(-1))
        completion_mask = (is_eos.long().cumsum(dim=1) - is_eos.long()) == 0

    sample_batch['on_policy_input_ids'] = generated_ids
    sample_batch['on_policy_attention_mask'] = torch.cat(
        [sample_batch['prompt_attention_mask'], completion_mask.long()], dim=1
    )
    sample_batch['completion_mask'] = completion_mask
    sample_batch['prompt_length'] = prompt_length

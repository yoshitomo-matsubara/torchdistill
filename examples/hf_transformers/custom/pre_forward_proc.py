import torch

from torchdistill.core.interfaces.pre_forward_proc import register_pre_forward_proc_func


@register_pre_forward_proc_func
def generate_on_policy_sequences(self, sample_batch=None, **generation_kwargs):
    """
    Collects on-policy data by letting the student generate completions for the prompts in ``sample_batch``.

    The generated sequences are written back into ``sample_batch`` so that the teacher and student forward
    processes score the same sequences, and so that the mid-level loss can restrict the KL divergence to the
    generated (completion) tokens:

    * ``on_policy_input_ids`` / ``on_policy_attention_mask``: prompt + generated completion.
    * ``completion_mask``: True at the generated positions that are not padding.
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
            'on-policy rollouts require prompts in the batch, '
            'which only the `instruction` preprocess pipeline provides'
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
    sample_batch['on_policy_input_ids'] = generated_ids
    sample_batch['on_policy_attention_mask'] = (generated_ids != pad_token_id).long()
    sample_batch['completion_mask'] = generated_ids[:, prompt_length:] != pad_token_id
    sample_batch['prompt_length'] = prompt_length

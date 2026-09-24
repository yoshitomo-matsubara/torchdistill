"""
On-policy knowledge distillation for Large Language Models (LLMs) using
torchdistill and Hugging Face transformers.

The student generates text sequences from prompts (on-policy data collection),
then the token-level KL divergence between the student and teacher distributions
over those sequences is minimized. This follows the GKD framework:

  Gu et al., "Generalized Knowledge Distillation for Auto-regressive Sequence Models"
  https://arxiv.org/abs/2306.13649

Modifications by Yoshitomo Matsubara, Copyright 2025.
"""
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import time

import datasets
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.backends import cudnn
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom.optim import customize_lr_config
from torchdistill.common import file_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, setup_for_distributed, set_seed, import_dependencies
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.datasets.registry import register_collate_func
from torchdistill.misc.log import set_basic_log_config, setup_log_file, SmoothedValue, MetricLogger

logger = def_logger.getChild(__name__)


def get_argparser():
    parser = argparse.ArgumentParser(description='On-policy knowledge distillation for LLMs')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--seed', type=int, default=None, help='a seed for reproducible training')
    parser.add_argument('-disable_cudnn_benchmark', action='store_true', help='disable torch.backends.cudnn.benchmark')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument(
        '-adjust_lr', action='store_true',
        help='multiply learning rate by number of distributed processes (world_size)'
    )
    return parser


def load_tokenizer_and_model(model_config, prioritizes_dst_ckpt=False):
    tokenizer_config = model_config['tokenizer_kwargs']
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    # Many causal LMs lack a dedicated pad token; reuse eos_token so padding works
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(model_config['model_kwargs'])
    if prioritizes_dst_ckpt and file_util.check_if_exists(model_config.get('dst_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['dst_ckpt']
    elif file_util.check_if_exists(model_config.get('src_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['src_ckpt']
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def preprocess_lm_datasets(
        raw_dataset_dict, tokenizer, prompt_key, response_key, max_prompt_length, max_length, base_split_name,
        batched, skipped_splits=None, dataset_id_map=None, use_chat_template=False, system_prompt=None, **kwargs
):
    """
    Tokenize an instruction-following dataset for on-policy LLM distillation.

    Each processed example contains:
    - ``input_ids`` / ``attention_mask`` / ``labels``: full (right-padded) sequence
      where prompt tokens are masked (-100) in ``labels`` for off-policy supervision.
    - ``prompt_input_ids`` / ``prompt_attention_mask``: prompt-only tokens
      (left-padded at collation time) used to seed on-policy generation.
    - ``prompt_length``: unpadded prompt token count, used to locate the
      completion boundary inside the generated sequence.
    """
    if skipped_splits is not None:
        for split in skipped_splits:
            raw_dataset_dict.pop(split, None)

    def _build_prompt_text(prompt):
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _build_full_text(prompt, response):
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            messages.append({'role': 'assistant', 'content': response})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return prompt + response

    def _preprocess(examples):
        prompt_texts = [_build_prompt_text(p) for p in examples[prompt_key]]
        full_texts = [_build_full_text(p, r)
                      for p, r in zip(examples[prompt_key], examples[response_key])]

        prompt_enc = tokenizer(
            prompt_texts,
            max_length=max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )
        full_enc = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        prompt_lengths = [len(ids) for ids in prompt_enc['input_ids']]

        # Mask prompt positions with -100 so the task loss only covers the completion
        labels = []
        for full_ids, plen in zip(full_enc['input_ids'], prompt_lengths):
            label = [-100] * plen + list(full_ids[plen:])
            labels.append(label)

        return {
            'input_ids': full_enc['input_ids'],
            'attention_mask': full_enc['attention_mask'],
            'labels': labels,
            'prompt_input_ids': prompt_enc['input_ids'],
            'prompt_attention_mask': prompt_enc['attention_mask'],
            'prompt_length': prompt_lengths,
        }

    remove_columns = raw_dataset_dict[base_split_name].column_names
    processed = raw_dataset_dict.map(
        _preprocess, batched=batched, remove_columns=remove_columns, **kwargs
    )

    if dataset_id_map is not None:
        result = {}
        for dataset_id, split_name in dataset_id_map.items():
            result[dataset_id] = processed[split_name]
        return result
    return processed


class OnPolicyDataCollator:
    """
    Collates variable-length sequences for on-policy LLM distillation.

    Full sequences are right-padded (standard for training).
    Prompt sequences are left-padded so that all prompts end at the same
    position in the tensor — a requirement for batched causal-LM generation.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        prompt_input_ids = [f['prompt_input_ids'] for f in features]
        prompt_attention_mask = [f['prompt_attention_mask'] for f in features]
        prompt_lengths = [f['prompt_length'] for f in features]

        max_len = max(len(ids) for ids in input_ids)
        max_prompt_len = max(len(ids) for ids in prompt_input_ids)

        padded_input_ids, padded_attn_mask, padded_labels = [], [], []
        for ids, mask, label in zip(input_ids, attention_mask, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attn_mask.append(mask + [0] * pad_len)
            padded_labels.append(label + [-100] * pad_len)

        # Left-pad prompts so all real tokens are flush with the right edge
        padded_prompt_ids, padded_prompt_mask = [], []
        for ids, mask in zip(prompt_input_ids, prompt_attention_mask):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append([self.pad_token_id] * pad_len + ids)
            padded_prompt_mask.append([0] * pad_len + mask)

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attn_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'prompt_input_ids': torch.tensor(padded_prompt_ids, dtype=torch.long),
            'prompt_attention_mask': torch.tensor(padded_prompt_mask, dtype=torch.long),
            'prompt_lengths': torch.tensor(prompt_lengths, dtype=torch.long),
        }


def compute_kl_loss(student_logits, teacher_logits, completion_ids, pad_token_id, kl_type):
    """
    Token-level KL divergence between student and teacher over completion tokens.

    ``student_logits`` and ``teacher_logits`` are shifted so that position *t*
    predicts ``completion_ids[:, t]``.  Padding positions are excluded from the loss.

    :param student_logits: student logits aligned with completion tokens, shape (B, T, V).
    :param teacher_logits: teacher logits aligned with completion tokens, shape (B, T, V).
    :param completion_ids: generated completion token ids, shape (B, T).
    :param pad_token_id: token id used for padding (excluded from loss).
    :param kl_type: ``'forward'`` for KL(p_T ∥ p_S) or ``'reverse'`` for KL(p_S ∥ p_T).
    :returns: scalar KL loss.
    """
    completion_mask = completion_ids != pad_token_id  # (B, T)
    if not completion_mask.any():
        return student_logits.sum() * 0.0  # differentiable zero

    student_log_probs = F.log_softmax(student_logits, dim=-1)   # (B, T, V)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)   # (B, T, V)

    # Flatten to (N, V) selecting only non-padding completion positions
    s_flat = student_log_probs[completion_mask]  # (N, V)
    t_flat = teacher_log_probs[completion_mask]  # (N, V)

    if kl_type == 'forward':
        # KL(p_T ∥ p_S): teacher-covering; gradient flows through student log-probs
        kl = F.kl_div(s_flat, t_flat.detach().exp(), reduction='batchmean')
    else:
        # KL(p_S ∥ p_T): mode-seeking; gradient flows through student probabilities
        kl = F.kl_div(t_flat.detach(), s_flat.exp(), reduction='batchmean')
    return kl


def on_policy_forward(
        batch, teacher_model, student_model, tokenizer, generation_kwargs, kl_type, alpha, accelerator
):
    """
    One on-policy distillation step.

    1. The student generates completions from prompts (no gradient).
    2. Teacher and student compute token logits over the generated sequences.
    3. KL divergence is computed on the generated (completion) tokens only.
    4. The total loss mixes the on-policy KL term with an off-policy task loss
       weighted by *alpha* (GKD formulation: ``(1-α)·KL + α·CE``).
    """
    prompt_input_ids = batch['prompt_input_ids']
    prompt_attention_mask = batch['prompt_attention_mask']
    pad_token_id = tokenizer.pad_token_id

    # --- 1. On-policy generation (no grad) ---
    # unwrap to access generate() regardless of DDP/FSDP wrapping
    raw_student = accelerator.unwrap_model(student_model)
    raw_student.eval()
    with torch.no_grad():
        generated_ids = raw_student.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            pad_token_id=pad_token_id,
            **generation_kwargs,
        )
    raw_student.train()

    # Because prompts were left-padded, the completion always begins at
    # the fixed offset max_prompt_len in the output tensor.
    max_prompt_len = prompt_input_ids.shape[1]
    completion_ids = generated_ids[:, max_prompt_len:]          # (B, max_new_tokens)
    gen_attention_mask = (generated_ids != pad_token_id).long() # (B, max_prompt_len + max_new_tokens)

    # --- 2. Teacher logits on generated sequences (no grad) ---
    with torch.no_grad():
        teacher_logits = teacher_model(
            input_ids=generated_ids,
            attention_mask=gen_attention_mask,
        ).logits  # (B, T, V)

    # --- 3. Student logits on generated sequences (with grad) ---
    student_logits = student_model(
        input_ids=generated_ids,
        attention_mask=gen_attention_mask,
    ).logits  # (B, T, V)

    # Shift logits to align with completion predictions:
    #   logits[:, max_prompt_len-1:-1, :] predicts completion_ids[:, 0:]
    kl_loss = compute_kl_loss(
        student_logits=student_logits[:, max_prompt_len - 1: -1, :],
        teacher_logits=teacher_logits[:, max_prompt_len - 1: -1, :],
        completion_ids=completion_ids,
        pad_token_id=pad_token_id,
        kl_type=kl_type,
    )

    if alpha > 0.0:
        # --- 4. Off-policy task loss (cross-entropy on reference completions) ---
        task_outputs = student_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return (1.0 - alpha) * kl_loss + alpha * task_outputs.loss

    return kl_loss


def train_one_epoch(
        training_box, teacher_model, student_model, tokenizer, epoch, log_freq, generation_kwargs, kl_type, alpha,
        accelerator
):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch in metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()

        if teacher_model is not None:
            loss = on_policy_forward(
                batch, teacher_model, student_model, tokenizer,
                generation_kwargs, kl_type, alpha, accelerator,
            )
        else:
            # Standard causal LM fine-tuning (no teacher)
            outputs = student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            loss = outputs.loss

        training_box.post_forward_process(loss=loss)

        batch_size = batch['input_ids'].shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))


@torch.inference_mode()
def evaluate(model, data_loader, accelerator, title=None, header='Test: '):
    """Computes per-token perplexity on *data_loader* (completion tokens only)."""
    if title is not None:
        logger.info(title)

    model.eval()
    total_nll = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0, device=accelerator.device)

    for batch in data_loader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        # outputs.loss is already mean-reduced over non-masked tokens within the batch
        num_tokens = (batch['labels'] != -100).sum()
        total_nll += outputs.loss * num_tokens
        total_tokens += num_tokens

    total_nll = accelerator.gather(total_nll).sum()
    total_tokens = accelerator.gather(total_tokens).sum()

    avg_nll = (total_nll / total_tokens).item() if total_tokens.item() > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    logger.info(f'{header}perplexity = {perplexity:.4f}, avg nll = {avg_nll:.4f}')
    return perplexity


def train(
        teacher_model, student_model, tokenizer, dataset_dict, dst_ckpt_dir_path, device, device_ids, distributed,
        config, args, accelerator
):
    logger.info('Start training')
    train_config = config['train']
    on_policy_config = train_config.get('on_policy', dict())
    kl_type = on_policy_config.get('kl_type', 'forward')
    alpha = float(on_policy_config.get('alpha', 0.5))
    generation_kwargs = on_policy_config.get('generation', dict())
    log_freq = train_config['log_freq']

    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    if teacher_model is None:
        training_box = get_training_box(
            student_model, dataset_dict, train_config,
            device, device_ids, distributed, lr_factor, accelerator,
        )
        # TrainingBox exposes the student as .model
        teacher_model = None
        student_model = training_box.model
    else:
        training_box = get_distillation_box(
            teacher_model, student_model, dataset_dict, train_config,
            device, device_ids, distributed, lr_factor, accelerator,
        )
        # DistillationBox exposes both models after accelerator.prepare()
        teacher_model = training_box.teacher_model
        student_model = training_box.student_model

    best_val_perplexity = float('inf')
    for epoch in range(training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        train_one_epoch(
            training_box, teacher_model, student_model, tokenizer,
            epoch, log_freq, generation_kwargs, kl_type, alpha, accelerator,
        )
        val_perplexity = evaluate(
            student_model, training_box.val_data_loader, accelerator, header='Validation: ',
        )
        if val_perplexity < best_val_perplexity:
            logger.info('Updating ckpt at {}'.format(dst_ckpt_dir_path))
            best_val_perplexity = val_perplexity
            accelerator.wait_for_everyone()
            # get_state_dict() performs the collective full-parameter gather FSDP/FSDP2 require;
            # a plain unwrapped_model.state_dict() would only capture the local shard.
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(
                dst_ckpt_dir_path, is_main_process=accelerator.is_main_process,
                save_function=accelerator.save, state_dict=accelerator.get_state_dict(student_model),
            )
        training_box.post_epoch_process()


def main(args):
    set_basic_log_config()
    if is_main_process() and args.run_log is not None:
        setup_log_file(os.path.expanduser(args.run_log))

    world_size = args.world_size
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True

    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    import_dependencies(config.get('dependencies', None))

    accelerator = Accelerator()
    # num_processes > 1 covers every multi-process backend Accelerate supports (DDP, FSDP,
    # DeepSpeed, etc.), whereas comparing against DistributedType.MULTI_GPU alone misses FSDP.
    distributed = accelerator.num_processes > 1
    device_ids = [accelerator.device.index]
    if distributed:
        setup_for_distributed(is_main_process())

    logger.info(accelerator.state)
    device = accelerator.device

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_tokenizer, teacher_model = (None, None) if teacher_model_config is None \
        else load_tokenizer_and_model(teacher_model_config, prioritizes_dst_ckpt=True)

    student_model_config = (
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    )
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, prioritizes_dst_ckpt=False)
    dst_ckpt_dir_path = student_model_config['dst_ckpt']

    # Set left-padding on the student tokenizer for batched generation
    student_tokenizer.padding_side = 'left'

    dataset_dict = preprocess_lm_datasets(
        config['datasets'], tokenizer=student_tokenizer, **config['preprocess']
    )

    customize_lr_config(config, dataset_dict, world_size)

    register_collate_func(OnPolicyDataCollator(pad_token_id=student_tokenizer.pad_token_id))

    if not args.test_only:
        train(
            teacher_model, student_model, student_tokenizer, dataset_dict,
            dst_ckpt_dir_path, device, device_ids, distributed, config, args, accelerator,
        )
        student_tokenizer.save_pretrained(dst_ckpt_dir_path)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(
        dataset_dict[test_data_loader_config['dataset_id']], test_data_loader_config, distributed
    )
    test_data_loader = accelerator.prepare(test_data_loader)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if not args.student_only and teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)
        evaluate(
            teacher_model, test_data_loader, accelerator, title='[Teacher: {}]'.format(teacher_model_config['key'])
        )

    # Reload best checkpoint for final evaluation
    _, student_model = load_tokenizer_and_model(student_model_config, prioritizes_dst_ckpt=True)
    student_model = accelerator.prepare(student_model)
    evaluate(
        student_model, test_data_loader, accelerator, title='[Student: {}]'.format(student_model_config['key'])
    )


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

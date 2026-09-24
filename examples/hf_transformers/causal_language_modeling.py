"""
Causal language modeling for Large Language Models (LLMs) using torchdistill and
Hugging Face transformers, with or without a teacher.

Two data pipelines are available, selected by `preprocess.key` in the yaml file:

- `instruction` (default): prompt/response data, where the prompt tokens are masked out of the loss.
  This is the pipeline on-policy distillation needs, since it is what supplies the prompts.
- `causal_lm`: a raw text corpus, concatenated and chunked into fixed-length blocks for pre-training.
  Every token is scored, and no prompts exist, so on-policy rollouts are not available here.

Both pipelines support plain training (no teacher) and off-policy distillation. On top of the
`instruction` pipeline, the student can also generate the sequences it is scored on (on-policy data
collection), minimizing the token-level KL divergence over those sequences. This follows the GKD framework:

  Gu et al., "Generalized Knowledge Distillation for Auto-regressive Sequence Models"
  https://arxiv.org/abs/2306.13649

The rollouts, the sequence scoring, and the KL divergence are all driven by the yaml configuration through
torchdistill's distillation box: `custom.pre_forward_proc` collects the rollouts, `custom.forward_proc`
scores them, and `custom.loss.OnPolicyKDLoss` / `custom.loss.OffPolicyKDLoss` are the mid-level losses,
combined with the task loss by `WeightedSumLoss`.

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
from torchdistill.misc.log import set_basic_log_config, setup_log_file, setup_tracker, SmoothedValue, MetricLogger

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
    parser.add_argument(
        '-disable_tracker', action='store_true',
        help='disable experiment tracker (trackio/wandb) even if configured in the yaml file'
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


def finalize_processed_datasets(processed_dataset_dict, dataset_id_map):
    """Maps split names to the dataset ids the yaml data loader configurations refer to."""
    if dataset_id_map is None:
        return processed_dataset_dict
    return {dataset_id: processed_dataset_dict[split_name] for dataset_id, split_name in dataset_id_map.items()}


def preprocess_instruction_datasets(
        raw_dataset_dict, tokenizer, prompt_key, response_key, max_prompt_length, max_length,  base_split_name, batched,
        skipped_splits=None, dataset_id_map=None, use_chat_template=False, system_prompt=None, **kwargs
):
    """
    Tokenize an instruction-following dataset for LLM training / distillation.

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
    return finalize_processed_datasets(processed, dataset_id_map)


def preprocess_causal_lm_datasets(
        raw_dataset_dict, tokenizer, text_key, block_size, base_split_name, skipped_splits=None,
        dataset_id_map=None, **kwargs
):
    """
    Tokenize a raw text corpus for causal language model pre-training.

    Documents are tokenized, joined by ``eos_token_id``, concatenated and then chunked into blocks of
    exactly ``block_size`` tokens, so that batches carry no padding and long documents are split across
    blocks rather than truncated. The trailing remainder that does not fill a block is dropped.

    Each processed example contains:
    - ``input_ids`` / ``attention_mask``: one full block, never padded.
    - ``labels``: a copy of ``input_ids``, i.e. every token is scored (no prompt masking).

    There are no prompts here, so on-policy rollouts are not available for this pipeline.
    """
    if skipped_splits is not None:
        for split in skipped_splits:
            raw_dataset_dict.pop(split, None)

    # Both maps below are necessarily batched, as blocks span document boundaries
    kwargs.pop('batched', None)
    eos_token_id = tokenizer.eos_token_id

    def _tokenize(examples):
        # Special tokens are added by hand below, so that documents are separated by exactly one eos token
        return tokenizer(examples[text_key], add_special_tokens=False)

    def _group_into_blocks(examples):
        concatenated = []
        for ids in examples['input_ids']:
            concatenated.extend(ids)
            concatenated.append(eos_token_id)

        num_blocks = len(concatenated) // block_size
        blocks = [concatenated[i * block_size: (i + 1) * block_size] for i in range(num_blocks)]
        return {
            'input_ids': blocks,
            'attention_mask': [[1] * block_size for _ in blocks],
            'labels': [list(block) for block in blocks],
        }

    remove_columns = raw_dataset_dict[base_split_name].column_names
    tokenized = raw_dataset_dict.map(_tokenize, batched=True, remove_columns=remove_columns, **kwargs)
    # Blocks span document boundaries, so this map is batched regardless of what the instruction pipeline does
    processed = tokenized.map(_group_into_blocks, batched=True, **kwargs)
    return finalize_processed_datasets(processed, dataset_id_map)


@register_collate_func
def collate_lm_blocks(features):
    """Stacks the fixed-length blocks produced by ``preprocess_causal_lm_datasets``. No padding is needed."""
    return {
        key: torch.tensor([f[key] for f in features], dtype=torch.long)
        for key in ('input_ids', 'attention_mask', 'labels')
    }


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


# Chosen by `preprocess.key` in the yaml file, defaulting to 'instruction' for backward compatibility
PREPROCESS_FUNC_DICT = {
    'instruction': preprocess_instruction_datasets,
    'causal_lm': preprocess_causal_lm_datasets,
}


def train_one_epoch(training_box, epoch, log_freq, tracker=None):
    metric_logger = MetricLogger(
        delimiter='  ', tracker=tracker, tracker_prefix='train/',
        tracker_start_step=epoch * len(training_box.train_data_loader)
    )
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch in metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        # Collects on-policy data and stores it in sample_batch. This is a no-op unless the training
        # configuration sets `pre_forward_process`, so the same loop covers plain fine-tuning.
        # Any `kwargs` given to `pre_forward_process` in the yaml file (e.g., `max_new_tokens`) are passed by the box
        training_box.pre_forward_process(sample_batch=sample_batch)
        # sample_batch doubles as the targets, as the criterion needs the on-policy metadata it carries
        loss = training_box.forward_process(sample_batch, targets=sample_batch, supp_dict=None)
        training_box.post_forward_process(loss=loss)
        batch_size = sample_batch['input_ids'].shape[0]
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
        teacher_model, student_model, dataset_dict, dst_ckpt_dir_path, device, device_ids, distributed,
        config, args, accelerator, tracker=None
):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(
        student_model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator
    ) if teacher_model is None else get_distillation_box(
        teacher_model, student_model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator
    )
    log_freq = train_config['log_freq']
    best_val_perplexity = float('inf')
    for epoch in range(training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        train_one_epoch(training_box, epoch, log_freq, tracker=tracker)
        val_perplexity = evaluate(student_model, training_box.val_data_loader, accelerator, header='Validation: ')
        if tracker is not None:
            tracker.log(
                {'val/perplexity': val_perplexity, 'epoch': epoch},
                step=(epoch + 1) * len(training_box.train_data_loader)
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

    preprocess_config = dict(config['preprocess'])
    preprocess_key = preprocess_config.pop('key', 'instruction')
    if preprocess_key == 'instruction':
        # Set left-padding on the student tokenizer for batched generation
        student_tokenizer.padding_side = 'left'
        register_collate_func(OnPolicyDataCollator(pad_token_id=student_tokenizer.pad_token_id))

    if preprocess_key not in PREPROCESS_FUNC_DICT:
        raise ValueError('preprocess key `{}` should be one of {}'.format(
            preprocess_key, sorted(PREPROCESS_FUNC_DICT.keys())))

    preprocess_func = PREPROCESS_FUNC_DICT[preprocess_key]
    dataset_dict = preprocess_func(config['datasets'], tokenizer=student_tokenizer, **preprocess_config)

    customize_lr_config(config, dataset_dict, world_size)

    tracker = setup_tracker(config.get('tracker', None), run_config=config) \
        if accelerator.is_main_process and not args.disable_tracker else None
    if not args.test_only:
        train(
            teacher_model, student_model, dataset_dict, dst_ckpt_dir_path,
            device, device_ids, distributed, config, args, accelerator, tracker=tracker,
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
    test_perplexity = evaluate(
        student_model, test_data_loader, accelerator, title='[Student: {}]'.format(student_model_config['key'])
    )
    if tracker is not None:
        tracker.log({'test/perplexity': test_perplexity})
        tracker.finish()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

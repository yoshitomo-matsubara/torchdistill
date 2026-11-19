"""
Abstractive summarization using torchdistill and Hugging Face transformers, with or without a teacher.

Both encoder-decoder (e.g., T5) and decoder-only (e.g., SmolLM2) models are supported. The architecture is read
from the checkpoint's config, so the yaml file needs no flag; the teacher and the student must agree.

- Encoder-decoder: the (prefixed) document is the encoder input and the summary is the decoder target.
- Decoder-only: the document is wrapped in a prompt template, the summary follows it in one sequence, and the
  prompt positions are masked out of the loss.

Plain fine-tuning (no teacher), off-policy distillation (`OffPolicyKDLoss`) and on-policy distillation
(`generate_on_policy_sequences` + `OnPolicyKDLoss`) are all driven by the yaml configuration, as in
causal_language_modeling.py. The sample configs reproduce the XSum experiments of GKD:

  Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes", ICLR 2024
  https://openreview.net/forum?id=3zKtaqxLhW

Summaries are evaluated by ROUGE-1/2/L (F1, with stemming) of greedy (or configured) generations.
"""

import argparse
import json
import logging
import math
import os
import shutil
import time

import datasets
import torch
import transformers
from accelerate import Accelerator
from rouge_score import rouge_scorer
from torch.backends import cudnn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from custom.dataset import OnPolicyDataCollator, Seq2SeqDataCollator
from custom.optim import customize_lr_config
from torchdistill.common import file_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, setup_for_distributed, set_seed, import_dependencies
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.datasets.registry import register_collate_func
from torchdistill.misc.log import set_basic_log_config, setup_log_file, setup_tracker, SmoothedValue, MetricLogger, \
    TrainingTrackerReader

logger = def_logger.getChild(__name__)

ROUGE_TYPES = ('rouge1', 'rouge2', 'rougeL')
DEFAULT_PROMPT_TEMPLATE = 'Summarize the following article in one sentence.\n\nArticle: {document}\n\nSummary:'


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for summarization models')
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


def resolve_model_path(model_config, prioritizes_dst_ckpt=False):
    if prioritizes_dst_ckpt and file_util.check_if_exists(model_config.get('dst_ckpt', None)):
        return model_config['dst_ckpt']
    if file_util.check_if_exists(model_config.get('src_ckpt', None)):
        return model_config['src_ckpt']
    return model_config['model_kwargs']['pretrained_model_name_or_path']


def is_encoder_decoder_model(model_config):
    return AutoConfig.from_pretrained(resolve_model_path(model_config)).is_encoder_decoder


def load_tokenizer_and_model(model_config, prioritizes_dst_ckpt=False):
    tokenizer = AutoTokenizer.from_pretrained(**model_config['tokenizer_kwargs'])
    # Many causal LMs lack a dedicated pad token; reuse eos_token so padding works
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(model_config['model_kwargs'])
    model_kwargs['pretrained_model_name_or_path'] = resolve_model_path(model_config, prioritizes_dst_ckpt)
    model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder_model(model_config) else AutoModelForCausalLM
    model = model_cls.from_pretrained(**model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    # Some Hub checkpoints store weights as transposed views (e.g., the pytorch_model.bin of google/t5-*-lm-adapt),
    # which loading keeps. FSDP2 cannot shard non-contiguous parameters; the values are unchanged
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    if model_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    return tokenizer, model


def alias_input_embedding_modules(model):
    # Transformers 5 gives T5's encoder and decoder their own embed_tokens modules that hold the same parameter as
    # `shared`. Accelerate's FSDP2 preparation shards the input embeddings (`shared`) as their own group, leaving
    # the same parameter in the root group through those modules, which FSDP2 rejects. Pointing the modules back
    # at `shared` (as in Transformers 4) keeps every use of the parameter in one group
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if child is not input_embeddings and isinstance(child, torch.nn.Embedding) \
                    and child.weight is input_embeddings.weight:
                setattr(module, child_name, input_embeddings)


def finalize_processed_datasets(processed_dataset_dict, dataset_id_map):
    if dataset_id_map is None:
        return processed_dataset_dict
    split_names = list(dataset_id_map.values())
    if len(set(split_names)) < len(split_names):
        raise ValueError(f'dataset_id_map maps multiple dataset ids to the same split: {dataset_id_map}')
    return {dataset_id: processed_dataset_dict[split_name] for dataset_id, split_name in dataset_id_map.items()}


def preprocess_seq2seq_datasets(
        raw_dataset_dict, tokenizer, document_key, summary_key, max_input_length, max_target_length,
        base_split_name, batched, input_prefix='', dataset_id_map=None, **kwargs
):
    def _preprocess(examples):
        model_inputs = tokenizer(
            [input_prefix + document for document in examples[document_key]],
            max_length=max_input_length, truncation=True
        )
        model_inputs['labels'] = tokenizer(
            text_target=examples[summary_key], max_length=max_target_length, truncation=True
        )['input_ids']
        # Untokenized references, so that ROUGE is not computed against truncated or re-detokenized summaries
        model_inputs['reference'] = examples[summary_key]
        return model_inputs

    remove_columns = raw_dataset_dict[base_split_name].column_names
    processed = {
        split_name: raw_dataset.map(_preprocess, batched=batched, remove_columns=remove_columns, **kwargs)
        for split_name, raw_dataset in raw_dataset_dict.items()
    }
    return finalize_processed_datasets(processed, dataset_id_map)


def preprocess_causal_lm_datasets(
        raw_dataset_dict, tokenizer, document_key, summary_key, max_input_length, max_target_length,
        base_split_name, batched, prompt_template=DEFAULT_PROMPT_TEMPLATE, use_chat_template=False,
        system_prompt=None, dataset_id_map=None, **kwargs
):
    # The template is tokenized around the document, so that only the document is truncated to fit the prompt
    # into max_input_length and the instruction and the answer cue always survive
    placeholder = '\0DOCUMENT\0'
    prompt_text = prompt_template.format(document=placeholder)
    if use_chat_template:
        messages = list()
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt_text})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    head_text, tail_text = prompt_text.split(placeholder)
    head_ids = tokenizer(head_text, add_special_tokens=True)['input_ids']
    tail_ids = tokenizer(tail_text, add_special_tokens=False)['input_ids']
    document_budget = max_input_length - len(head_ids) - len(tail_ids)
    if document_budget <= 0:
        raise ValueError(f'max_input_length `{max_input_length}` leaves no room for the document in the prompt')

    def _preprocess(examples):
        document_ids_list = tokenizer(examples[document_key], add_special_tokens=False)['input_ids']
        summary_ids_list = tokenizer(examples[summary_key], add_special_tokens=False)['input_ids']
        input_ids_list, labels_list, prompt_ids_list = list(), list(), list()
        for document_ids, summary_ids in zip(document_ids_list, summary_ids_list):
            prompt_ids = head_ids + document_ids[:document_budget] + tail_ids
            # The completion ends with EOS so that the model learns to stop
            completion_ids = summary_ids[:max_target_length - 1] + [tokenizer.eos_token_id]
            input_ids_list.append(prompt_ids + completion_ids)
            labels_list.append([-100] * len(prompt_ids) + completion_ids)
            prompt_ids_list.append(prompt_ids)
        return {
            'input_ids': input_ids_list,
            'attention_mask': [[1] * len(ids) for ids in input_ids_list],
            'labels': labels_list,
            'prompt_input_ids': prompt_ids_list,
            'prompt_attention_mask': [[1] * len(ids) for ids in prompt_ids_list],
            'prompt_length': [len(ids) for ids in prompt_ids_list],
            'reference': examples[summary_key],
        }

    remove_columns = raw_dataset_dict[base_split_name].column_names
    processed = {
        split_name: raw_dataset.map(_preprocess, batched=batched, remove_columns=remove_columns, **kwargs)
        for split_name, raw_dataset in raw_dataset_dict.items()
    }
    return finalize_processed_datasets(processed, dataset_id_map)


class ReferenceCollator:
    """Passes the untokenized reference summaries through as a list next to the collated tensors."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        references = [f.pop('reference') for f in features]
        batch = self.collator(features)
        batch['references'] = references
        return batch


def train_one_epoch(training_box, epoch, log_freq, max_steps=None, tracker=None):
    metric_logger = MetricLogger(
        delimiter='  ', tracker=tracker, tracker_prefix='train/',
        tracker_start_step=epoch * len(training_box.train_data_loader)
    )
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch in metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        # No-op unless the training configuration sets `pre_forward_process` (e.g., on-policy rollouts)
        training_box.pre_forward_process(sample_batch=sample_batch)
        # sample_batch doubles as the targets, as the criterion needs the on-policy metadata it carries
        loss = training_box.forward_process(sample_batch, targets=sample_batch, supp_dict=None)
        training_box.post_forward_process(loss=loss)
        batch_size = sample_batch['input_ids'].shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))
        if max_steps is not None and count_optimizer_steps(training_box) >= max_steps:
            return True
    return False


def count_optimizer_steps(training_box):
    return training_box.stage_grad_count // training_box.grad_accum_step


@torch.inference_mode()
def evaluate_perplexity(model, data_loader, accelerator, header='Validation: '):
    model.eval()
    total_nll = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0, device=accelerator.device)
    for batch in data_loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # outputs.loss is mean-reduced over the non-masked tokens of the batch
        num_tokens = (batch['labels'] != -100).sum()
        total_nll += outputs.loss * num_tokens
        total_tokens += num_tokens

    total_nll = accelerator.gather(total_nll).sum()
    total_tokens = accelerator.gather(total_tokens).sum()
    avg_nll = (total_nll / total_tokens).item() if total_tokens.item() > 0 else float('inf')
    perplexity = math.exp(avg_nll) if avg_nll < 100 else float('inf')
    logger.info(f'{header}perplexity = {perplexity:.4f}, avg nll = {avg_nll:.4f}')
    return perplexity


@torch.inference_mode()
def evaluate_rouge(model, tokenizer, data_loader, accelerator, generation_kwargs, title=None, header='Test: '):
    if title is not None:
        logger.info(title)

    model.eval()
    is_encoder_decoder = model.config.is_encoder_decoder
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    score_list = list()
    for batch in data_loader:
        prompt_input_ids = batch['prompt_input_ids']
        with accelerator.autocast():
            generated_ids = model.generate(
                input_ids=prompt_input_ids, attention_mask=batch['prompt_attention_mask'],
                pad_token_id=tokenizer.pad_token_id, **generation_kwargs
            )
        # A decoder-only model echoes the left-padded prompt, which ends at the same column for every sample
        if not is_encoder_decoder:
            generated_ids = generated_ids[:, prompt_input_ids.shape[1]:]
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for prediction, reference in zip(predictions, batch['references']):
            scores = scorer.score(reference.strip(), prediction.strip())
            score_list.append([scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES])

    # gather_for_metrics drops the samples the distributed sampler duplicated to even out the processes
    score_tensor = accelerator.gather_for_metrics(torch.tensor(score_list, device=accelerator.device))
    mean_scores = (score_tensor.mean(dim=0) * 100).tolist()
    result = dict(zip(ROUGE_TYPES, mean_scores))
    logger.info(header + ', '.join(f'{key} = {value:.2f}' for key, value in result.items())
                + f' ({score_tensor.shape[0]} samples)')
    return result


def save_model(prepared_model, dst_ckpt_dir_path, accelerator):
    accelerator.wait_for_everyone()
    # get_state_dict() performs the collective full-parameter gather FSDP/FSDP2 require
    unwrapped_model = accelerator.unwrap_model(prepared_model)
    unwrapped_model.save_pretrained(
        dst_ckpt_dir_path, is_main_process=accelerator.is_main_process,
        save_function=accelerator.save, state_dict=accelerator.get_state_dict(prepared_model),
    )
    # Only the main process writes; the others must not go on to reload the checkpoint before it is complete
    accelerator.wait_for_everyone()


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
    # Accelerator.prepare() may wrap the model (e.g., FSDP), so evaluate and save through the box's prepared model
    prepared_model = training_box.model if teacher_model is None else training_box.student_model
    log_freq = train_config['log_freq']
    max_steps = train_config.get('max_steps', None)
    if max_steps is not None:
        num_iterations_per_epoch = len(training_box.train_data_loader)
        required_num_epochs = math.ceil(max_steps * training_box.grad_accum_step / num_iterations_per_epoch)
        if training_box.num_epochs < required_num_epochs:
            raise ValueError(
                f'num_epochs `{training_box.num_epochs}` is too small to reach max_steps `{max_steps}` with '
                f'{num_iterations_per_epoch} iterations per epoch; set num_epochs >= {required_num_epochs}'
            )

    # 'last' keeps the final checkpoint, as the paper reports; 'best' selects on the monitoring split
    ckpt_selection = train_config.get('ckpt_selection', 'best')
    if ckpt_selection not in ('best', 'last'):
        raise ValueError(f'ckpt_selection `{ckpt_selection}` should be either `best` or `last`')

    best_val_perplexity = float('inf')
    best_epoch = None
    num_epochs_run = 0
    for epoch in range(training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        reached_max_steps = train_one_epoch(training_box, epoch, log_freq, max_steps=max_steps, tracker=tracker)
        num_epochs_run = epoch + 1
        val_perplexity = evaluate_perplexity(prepared_model, training_box.val_data_loader, accelerator)
        if tracker is not None:
            tracker.log(
                {'val/perplexity': val_perplexity, 'epoch': epoch},
                step=(epoch + 1) * len(training_box.train_data_loader)
            )

        if ckpt_selection == 'best' and val_perplexity < best_val_perplexity:
            logger.info('Updating ckpt at {}'.format(dst_ckpt_dir_path))
            best_val_perplexity = val_perplexity
            best_epoch = epoch
            save_model(prepared_model, dst_ckpt_dir_path, accelerator)
        training_box.post_epoch_process()
        if reached_max_steps:
            logger.info(f'Reached max_steps `{max_steps}` in epoch {epoch}')
            break

    if ckpt_selection == 'last':
        logger.info('Saving the last ckpt at {}'.format(dst_ckpt_dir_path))
        save_model(prepared_model, dst_ckpt_dir_path, accelerator)
    return {
        'best_val_perplexity': best_val_perplexity if ckpt_selection == 'best' else None,
        'best_epoch': best_epoch,
        'num_epochs_run': num_epochs_run,
        'num_optimizer_steps': count_optimizer_steps(training_box),
    }


def save_run_settings(logs_dir_path, config_file_path, args):
    file_util.make_dirs(logs_dir_path)
    # Copy the yaml file verbatim, keeping its anchors and tags, rather than dumping the resolved config
    shutil.copyfile(config_file_path, os.path.join(logs_dir_path, 'config.yaml'))
    with open(os.path.join(logs_dir_path, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=2)


def save_metrics(logs_dir_path, metric_dict):
    with open(os.path.join(logs_dir_path, 'metrics.json'), 'w') as fp:
        json.dump(metric_dict, fp, indent=2)


def export_tracker_run(tracker, logs_dir_path):
    tracker_dir_path = os.path.join(logs_dir_path, tracker.engine)
    file_util.make_dirs(tracker_dir_path)
    project, run_name = tracker.run.project, tracker.run.name
    with open(os.path.join(tracker_dir_path, 'run_manifest.json'), 'w') as fp:
        json.dump({'engine': tracker.engine, 'project': project, 'run_name': run_name}, fp, indent=2)

    # The run itself already lives in the tracker's own storage, so a failed export only costs this local copy
    try:
        history = TrainingTrackerReader(tracker.engine).load_run_history(project, run_name)
        history.to_csv(os.path.join(tracker_dir_path, 'summary.csv'), index=False)
    except Exception as e:
        logger.warning(f'Failed to export the {tracker.engine} run history of `{project}/{run_name}`: {e}')


def main(args):
    set_basic_log_config()
    if is_main_process() and args.run_log is not None:
        setup_log_file(os.path.expanduser(args.run_log))

    world_size = args.world_size
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True

    set_seed(args.seed)
    config_file_path = os.path.expanduser(args.config)
    config = yaml_util.load_yaml_file(config_file_path)
    import_dependencies(config.get('dependencies', None))

    accelerator = Accelerator()
    # num_processes > 1 covers every multi-process backend Accelerate supports (DDP, FSDP, DeepSpeed, etc.)
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
    student_model_config = (
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    )
    is_encoder_decoder = is_encoder_decoder_model(student_model_config)
    if teacher_model_config is not None and is_encoder_decoder_model(teacher_model_config) != is_encoder_decoder:
        raise ValueError('The teacher and the student should be both encoder-decoder or both decoder-only models')

    teacher_tokenizer, teacher_model = (None, None) if teacher_model_config is None \
        else load_tokenizer_and_model(teacher_model_config, prioritizes_dst_ckpt=True)
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, prioritizes_dst_ckpt=False)
    if accelerator.is_fsdp2:
        alias_input_embedding_modules(student_model)
    dst_ckpt_dir_path = student_model_config['dst_ckpt']

    org_padding_side = student_tokenizer.padding_side
    if is_encoder_decoder:
        collator = Seq2SeqDataCollator(pad_token_id=student_tokenizer.pad_token_id)
        preprocess_func = preprocess_seq2seq_datasets
    else:
        # Batched generation with a decoder-only model needs left-padded prompts
        student_tokenizer.padding_side = 'left'
        collator = OnPolicyDataCollator(pad_token_id=student_tokenizer.pad_token_id)
        preprocess_func = preprocess_causal_lm_datasets

    # Registered under the base collator's name (e.g., 'Seq2SeqDataCollator'), which the yaml files refer to
    register_collate_func(key=type(collator).__name__)(ReferenceCollator(collator))
    dataset_dict = preprocess_func(config['datasets'], tokenizer=student_tokenizer, **config['preprocess'])
    train_config = config.get('train', None)
    if train_config is None or 'max_steps' not in train_config:
        customize_lr_config(config, dataset_dict, world_size)

    tracker = setup_tracker(config.get('tracker', None), run_config=config) \
        if accelerator.is_main_process and not args.disable_tracker else None
    logs_dir_path = os.path.join(dst_ckpt_dir_path, 'logs')
    metric_dict = dict()
    test_only = args.test_only
    if not test_only:
        if accelerator.is_main_process:
            save_run_settings(logs_dir_path, config_file_path, args)
        metric_dict.update(train(
            teacher_model, student_model, dataset_dict, dst_ckpt_dir_path,
            device, device_ids, distributed, config, args, accelerator, tracker=tracker,
        ))
        # Release the training state (e.g., FSDP shards and optimizer states) the accelerator still references,
        # or it stays on the GPUs next to the full student reloaded below for the final evaluation
        del student_model
        accelerator.free_memory()
        if student_tokenizer.is_fast:
            # Preprocessing with truncation=True leaves truncation enabled on the backend tokenizer,
            # which save_pretrained would otherwise persist into tokenizer.json
            student_tokenizer.backend_tokenizer.no_truncation()
        # Left-padding is only for batched generation; don't ship it with the checkpoint
        student_tokenizer.padding_side = org_padding_side
        student_tokenizer.save_pretrained(dst_ckpt_dir_path)

    test_config = config['test']
    generation_kwargs = test_config.get('generation_kwargs', dict())
    test_data_loader_dict = {
        name: accelerator.prepare(
            # Passing the accelerator keeps build_data_loader from adding a DistributedSampler: prepare() shards the
            # loader itself, and sharding twice would leave each process 1/N^2 of the samples
            util.build_data_loader(
                dataset_dict[loader_config['dataset_id']], loader_config, distributed, accelerator=accelerator
            )
        )
        for name, loader_config in test_config['test_data_loaders'].items()
    }
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Generation runs on plain (unwrapped) models: every process holds a full copy and the data loaders shard
    # the samples, which also avoids generating through FSDP-sharded parameters
    if not args.student_only and teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)
        for name, test_data_loader in test_data_loader_dict.items():
            metric_dict[f'teacher_{name}'] = evaluate_rouge(
                teacher_model, teacher_tokenizer, test_data_loader, accelerator, generation_kwargs,
                title='[Teacher: {}]'.format(teacher_model_config['key']), header=f'{name}: '
            )
        del teacher_model

    # Reload the saved checkpoint for the final evaluation
    _, student_model = load_tokenizer_and_model(student_model_config, prioritizes_dst_ckpt=True)
    student_model = student_model.to(accelerator.device)
    for name, test_data_loader in test_data_loader_dict.items():
        result = evaluate_rouge(
            student_model, student_tokenizer, test_data_loader, accelerator, generation_kwargs,
            title='[Student: {}]'.format(student_model_config['key']), header=f'{name}: '
        )
        metric_dict[f'student_{name}'] = result
        if tracker is not None:
            tracker.log({f'{name}/{key}': value for key, value in result.items()})

    if tracker is not None:
        tracker.finish()

    # A test-only run leaves logs/ alone so it does not overwrite the record of the run that trained the ckpt
    if not test_only and accelerator.is_main_process:
        save_metrics(logs_dir_path, metric_dict)
        if tracker is not None:
            export_tracker_run(tracker, logs_dir_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

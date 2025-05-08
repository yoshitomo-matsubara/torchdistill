"""
Example code to fine-tuning the Transformer models for sequence classification
at https://github.com/huggingface/transformers/blob/v4.49.0/examples/pytorch/text-classification/run_classification.py
modified to collaborate with torchdistill.

Original copyright of The HuggingFace Inc. code below, modifications by Yoshitomo Matsubara, Copyright 2025.
"""
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
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
from accelerate import Accelerator, DistributedType
from torch.backends import cudnn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
    PretrainedConfig

from custom.dataset import preprocess_hf_text_datasets
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
    parser = argparse.ArgumentParser(description='Knowledge distillation for text classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--task_name', type=str, default=None, help='name of the task for fine-tuning')
    parser.add_argument('--seed', type=int, default=None, help='a seed for reproducible training')
    parser.add_argument('-disable_cudnn_benchmark', action='store_true', help='disable torch.backend.cudnn.benchmark')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def load_tokenizer_and_model(model_config, task_name, prioritizes_dst_ckpt=False):
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_config = model_config['config_kwargs']
    config = AutoConfig.from_pretrained(**config_config, finetuning_task=task_name)
    tokenizer_config = model_config['tokenizer_kwargs']
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    model_kwargs = model_config['model_kwargs']
    if prioritizes_dst_ckpt and file_util.check_if_exists(model_config.get('dst_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['dst_ckpt']
    elif file_util.check_if_exists(model_config.get('src_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['src_ckpt']
    model = AutoModelForSequenceClassification.from_pretrained(config=config, **model_kwargs)
    return tokenizer, model


def train_one_epoch(training_box, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        loss = training_box.forward_process(sample_batch, targets=None, supp_dict=None)
        training_box.post_forward_process(loss=loss)
        batch_size = len(sample_batch)
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))


@torch.inference_mode()
def evaluate(model, data_loader, metric, evaluate_config, accelerator, title=None, header='Test: '):
    if title is not None:
        logger.info(title)

    uses_argmax = evaluate_config.get('argmax', True)
    thresholds = evaluate_config.get('thresholds', None)
    if isinstance(thresholds, (list, tuple)):
        thresholds = torch.Tensor(thresholds)

    add_batch_kwargs = evaluate_config.get('add_batch', dict())
    model.eval()
    for batch in data_loader:
        labels = batch.pop('labels')
        outputs = model(**batch)
        if uses_argmax:
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(labels),
                **add_batch_kwargs
            )
        elif thresholds is not None:
            predictions = (outputs.logits > thresholds).int()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(labels),
                **add_batch_kwargs
            )
        else:
            metric.add_batch(
                prediction_scores=accelerator.gather(outputs.logits),
                references=accelerator.gather(labels),
                **add_batch_kwargs
            )

    compute_kwargs = evaluate_config.get('compute', dict())
    eval_dict = metric.compute(**compute_kwargs)
    eval_desc = header + ', '.join([f'{key} = {eval_dict[key]}' for key in sorted(eval_dict.keys())])
    logger.info(eval_desc)
    return eval_dict


def train(teacher_model, student_model, dataset_dict, dst_ckpt_dir_path, metric, evaluate_config,
          device, device_ids, distributed, config, args, accelerator):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor, accelerator) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor, accelerator)
    # Only show the progress bar once on each machine.
    log_freq = train_config['log_freq']
    best_val_number = 0.0
    for epoch in range(training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        train_one_epoch(training_box, epoch, log_freq)
        val_dict = evaluate(student_model, training_box.val_data_loader, metric, evaluate_config, accelerator, header='Validation: ')
        val_value = sum(val_dict.values())
        if val_value > best_val_number:
            logger.info('Updating ckpt at {}'.format(dst_ckpt_dir_path))
            best_val_number = val_value
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(dst_ckpt_dir_path, save_function=accelerator.save)
        training_box.post_epoch_process()


def main(args):
    set_basic_log_config()
    log_file_path = args.run_log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    world_size = args.world_size
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True

    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    import_dependencies(config.get('dependencies', None))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    distributed = accelerator.state.distributed_type == DistributedType.MULTI_GPU
    device_ids = [accelerator.device.index]
    if distributed:
        setup_for_distributed(is_main_process())

    logger.info(accelerator.state)
    device = accelerator.device

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Load pretrained model and tokenizer
    task_name = args.task_name
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_tokenizer, teacher_model = (None, None) if teacher_model_config is None \
        else load_tokenizer_and_model(teacher_model_config, task_name, True)
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, task_name, False)
    dst_ckpt_dir_path = student_model_config['dst_ckpt']

    # Get datasets
    dataset_dict = preprocess_hf_text_datasets(config['datasets'], tokenizer=student_tokenizer, **config['preprocess'])

    # Update config with dataset size len(data_loader)
    customize_lr_config(config, dataset_dict, world_size)

    # register collate function
    register_collate_func(
        DataCollatorWithPadding(
            student_tokenizer,
            pad_to_multiple_of=16 if accelerator.mixed_precision == 'fp8' else
            8 if accelerator.mixed_precision != 'no' else None
        )
    )

    # Get the metric function
    metric = config['metric']
    evaluate_config = config['evaluate']
    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, dst_ckpt_dir_path, metric, evaluate_config,
              device, device_ids, distributed, config, args, accelerator)
        student_tokenizer.save_pretrained(dst_ckpt_dir_path)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    test_data_loader = accelerator.prepare(test_data_loader)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if not args.student_only and teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)
        evaluate(teacher_model, test_data_loader, metric, evaluate_config, accelerator,
                 title='[Teacher: {}]'.format(teacher_model_config['key']))

    # Reload the best checkpoint based on validation result
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, task_name, True)
    student_model = accelerator.prepare(student_model)
    evaluate(student_model, test_data_loader, metric, evaluate_config, accelerator,
             title='[Student: {}]'.format(student_model_config['key']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())

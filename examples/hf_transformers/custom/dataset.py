import os
import time

import torch
from datasets import load_dataset
from torch import distributed as dist
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process
from torchdistill.datasets.collator import register_collate_func
from torchdistill.datasets.registry import register_dataset
from transformers import squad_convert_examples_to_features, PretrainedConfig, default_data_collator
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

logger = def_logger.getChild(__name__)

GLUE_TASK2KEYS = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

register_collate_func(default_data_collator)


def load_raw_glue_datasets_and_misc(task_name, train_file_path=None, valid_file_path=None):
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset('glue', task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if train_file_path is not None:
            data_files['train'] = train_file_path
        if valid_file_path is not None:
            data_files['validation'] = valid_file_path
        extension = (train_file_path if train_file_path is not None else valid_file_path).split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_list = None
    if task_name is not None:
        is_regression = task_name == 'stsb'
        if not is_regression:
            label_list = raw_datasets['train'].features['label'].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets['train'].features['label'].dtype in ['float32', 'float64']
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets['train'].unique('label')
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return raw_datasets, num_labels, label_list, is_regression


def preprocess_glue_datasets(task_name, raw_datasets, num_labels, label_list, is_regression,
                             pad_to_max_length, max_length, tokenizer, model):
    # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = GLUE_TASK2KEYS[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets['train'].column_names if name != 'label']
        if 'sentence1' in non_label_column_names and 'sentence2' in non_label_column_names:
            sentence1_key, sentence2_key = 'sentence1', 'sentence2'
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f'The configuration of the model provided the following label correspondence: {label_name_to_id}. '
                'Using it!'
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f'model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}.'
                '\nIgnoring the model labels as a result.',
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = 'max_length' if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if 'label' in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result['labels'] = [label_to_id[l] for l in examples['label']]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result['labels'] = examples['label']
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names
    )
    return processed_datasets







@register_dataset
def load_and_cache_examples(input_file_path, cached_features_file_path, tokenizer, ver2_with_neg,
                            max_seq_length, doc_stride, max_query_length, num_threads,
                            training=False, overwrites_cache=False, output_examples=False):
    if not is_main_process() and training:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        dist.barrier()

    # Load data features from cache or dataset file
    if os.path.exists(cached_features_file_path) and not overwrites_cache:
        logger.info(f'Loading features from cached file {cached_features_file_path}')
        features_and_dataset = torch.load(cached_features_file_path)
        try:
            features, dataset, examples = (
                features_and_dataset['features'],
                features_and_dataset['dataset'],
                features_and_dataset['examples']
            )
        except KeyError:
            raise DeprecationWarning(
                f'You seem to be loading features from an older version of this script please delete the '
                f'file {cached_features_file_path} in order for it to be created again'
            )
    else:
        logger.info(f'Creating features from dataset file at {input_file_path}')
        processor = SquadV2Processor() if ver2_with_neg else SquadV1Processor()
        root_dir_path = os.path.dirname(input_file_path)
        file_name = os.path.basename(input_file_path)
        if training:
            examples = processor.get_train_examples(root_dir_path, filename=file_name)
        else:
            examples = processor.get_dev_examples(root_dir_path, filename=file_name)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=training,
            return_dataset="pt",
            threads=num_threads,
        )

        if is_main_process():
            logger.info(f'Saving features into cached file {cached_features_file_path}')
            torch.save({'features': features, 'dataset': dataset, 'examples': examples}, cached_features_file_path)

    if is_main_process() and training:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        dist.barrier()

    if output_examples:
        setattr(dataset, 'examples', examples)
        setattr(dataset, 'features', features)
        # return dataset, examples, features
    return dataset


def get_squad_style_dataset_dict(dataset_config, tokenizer):
    dataset_dict = dict()
    dataset_splits_config = dataset_config['splits']
    for split_name in dataset_splits_config.keys():
        st = time.time()
        logger.info('Loading {} data'.format(split_name))
        split_config = dataset_splits_config[split_name]
        org_dataset = load_and_cache_examples(tokenizer=tokenizer, **split_config['params'])
        dataset_id = split_config['dataset_id']
        dataset_dict[dataset_id] = org_dataset
        logger.info('{} sec'.format(time.time() - st))
    return dataset_dict


def get_all_squad_style_datasets(datasets_config, tokenizer):
    dataset_dict = dict()
    for dataset_name in datasets_config.keys():
        sub_dataset_dict = get_squad_style_dataset_dict(datasets_config[dataset_name], tokenizer)
        dataset_dict.update(sub_dataset_dict)
    return dataset_dict


def load_train_dataset(tokenizer, max_seq_length, pad_on_right, question_column_name, context_column_name,
                       answer_column_name, column_names, args, raw_datasets):

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")

    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))
    # Create train feature from dataset
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
    )
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset

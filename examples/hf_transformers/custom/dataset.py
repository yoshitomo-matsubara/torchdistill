from datasets import load_dataset
from torchdistill.common.constant import def_logger
from torchdistill.datasets.collator import register_collate_func
from transformers import PretrainedConfig, default_data_collator

logger = def_logger.getChild(__name__)

GLUE_TASK2KEYS = {
    'ax': ('premise', 'hypothesis'),
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


def load_raw_glue_datasets_and_misc(task_name, train_file_path=None, valid_file_path=None, base_split_name='train'):
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
            label_list = raw_datasets[base_split_name].features['label'].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets[base_split_name].features['label'].dtype in ['float32', 'float64']
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets[base_split_name].unique('label')
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return raw_datasets, num_labels, label_list, is_regression


def preprocess_glue_datasets(task_name, raw_datasets, num_labels, label_list, is_regression,
                             pad_to_max_length, max_length, tokenizer, model, base_split_name='train'):
    # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = GLUE_TASK2KEYS[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets[base_split_name].column_names if name != 'label']
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
        preprocess_function, batched=True, remove_columns=raw_datasets[base_split_name].column_names
    )
    return processed_datasets

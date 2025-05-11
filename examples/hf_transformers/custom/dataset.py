from transformers import default_data_collator

from torchdistill.common.constant import def_logger
from torchdistill.datasets.registry import register_collate_func

logger = def_logger.getChild(__name__)
register_collate_func(default_data_collator)


def extract_label_names(raw_dataset, label_key='label'):
    if isinstance(raw_dataset[label_key][0], list):
        label_names = [label for sample in raw_dataset[label_key] for label in sample]
        label_names = list(set(label_names))
    else:
        label_names = raw_dataset.unique(label_key)
    return label_names


def preprocess_hf_text_datasets(raw_dataset_dict, tokenizer, text_keys, label_key, pad_to_max_length,
                                max_length, base_split_name, batched, skipped_splits=None, remove_columns=None,
                                label2id=None, dataset_id_map=None, **kwargs):
    if skipped_splits is not None:
        for skipped_split in skipped_splits:
            raw_dataset_dict.pop(skipped_split)

    padding = 'max_length' if pad_to_max_length else False
    is_multi_label = False
    for key in raw_dataset_dict.keys():
        raw_dataset_dict[key] = raw_dataset_dict[key].rename_column(label_key, 'label')

    if 'label' in raw_dataset_dict[base_split_name].features:
        is_multi_label = raw_dataset_dict[base_split_name].features['label'].dtype == 'list'
        if label2id is None:
            label_names = extract_label_names(raw_dataset_dict[base_split_name])
            label_name_set = set(label_names)
            for split_name, raw_dataset in raw_dataset_dict.items():
                if split_name == base_split_name:
                    continue
                label_names = extract_label_names(raw_dataset)
                diff = set(label_names).difference(label_name_set)
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f'Labels {diff} in `{split_name}` split but not in `{base_split_name}` split, '
                        f'adding them to the label list'
                    )
                    label_name_set.update(diff)

            label_names = list(label_name_set)
            label_names.sort()
            label2id = {v: i for i, v in enumerate(label_names)}

    def _multi_labels_to_ids(labels):
        ids = [0.0] * len(label2id)
        for label in labels:
            ids[label2id[label]] = 1.0
        return ids

    def _preprocess(examples):
        texts = [examples[text_key] for text_key in text_keys]
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        if label2id is not None and 'label' in examples:
            if is_multi_label:
                result['label'] = [_multi_labels_to_ids(l) for l in examples['label']]
            else:
                result['label'] = [(label2id[str(l)] if l != -1 else -1) for l in examples['label']]
        return result

    if remove_columns is None:
        remove_columns = raw_dataset_dict[base_split_name].column_names

    for raw_dataset in raw_dataset_dict.values():
        if is_multi_label and raw_dataset.features['label'].feature.dtype.startswith('int'):
            raw_dataset.features['label'].feature.dtype = raw_dataset.features['label'].feature.dtype.replace('int', 'float')

    processed_datasets = raw_dataset_dict.map(_preprocess, batched=batched, remove_columns=remove_columns, **kwargs)
    if dataset_id_map is not None:
        dataset_dict = dict()
        for dataset_id, split_name in dataset_id_map.items():
            dataset_dict[dataset_id] = processed_datasets[split_name]
        return dataset_dict
    return processed_datasets

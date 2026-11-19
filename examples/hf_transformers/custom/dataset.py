import torch
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


def preprocess_hf_text_datasets(
        raw_dataset_dict, tokenizer, text_keys, label_key, pad_to_max_length, max_length, base_split_name,
        batched, skipped_splits=None, remove_columns=None, label2id=None, dataset_id_map=None, **kwargs
):
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


class Seq2SeqDataCollator:
    """
    Collates variable-length sequences for encoder-decoder models.

    Inputs and labels are right-padded. Encoder-decoder generation needs no left padding, so the padded inputs
    double as the prompts that seed on-policy generation.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_input_len = max(len(f['input_ids']) for f in features)
        max_label_len = max(len(f['labels']) for f in features)
        input_ids, attention_mask, labels = list(), list(), list()
        for f in features:
            pad_len = max_input_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.pad_token_id] * pad_len)
            attention_mask.append(f['attention_mask'] + [0] * pad_len)
            labels.append(f['labels'] + [-100] * (max_label_len - len(f['labels'])))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long),
            'prompt_input_ids': input_ids,
            'prompt_attention_mask': attention_mask,
        }

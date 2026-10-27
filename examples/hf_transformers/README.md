# Examples: Hugging Face Transformers

## 0. Requirements
### Install additional packages
You will first need to install Hugging Face's `transformers`, `datasets`, and `accelerate` packages.
Install dependencies using [requirements.txt](requirements.txt).  
```shell
pip install -r examples/hf_transformers/requirements.txt
```

Unless you use your own (local) datasets, datasets will be automatically downloaded by `datasets` package 
when running the following scripts.

### Configure accelerate
Accelerate provides an easy API for multi-GPUs/TPU/fp16 (mix-precision).  
```shell
accelerate config
```

## 1. Text classification tasks

### Fine-tuning a Transformer model
e.g., Fine-tuning BERT-Base (uncased) on GoEmotions dataset (multi-label classification)
```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/sample/go_emotions/ce/bert_base_uncased.yaml \
  --run_log log/go_emotions/ce/bert_base_uncased.txt \
  --seed 123 \
  -disable_cudnn_benchmark
```

## 2. GLUE tasks
GLUE consists of 9 different tasks: CoLA, SST-2, MRPC, STS-B (regression), QQP, MNLI, QNLI, RTE, and WNLI.  
For these tasks, you can run experiments with `cola`, `sst2`, `mrpc`, `stsb`, `qqp`, `mnli`, `qnli`, `rte`, and `wnli`, respectively.  
Note that STS-B (TASK_NAME=`stsb`) is a regression task, and the sample config file is under `mse/` instead of `ce/`.

### Fine-tuning a Transformer model
e.g., Fine-tuning BERT-Base (uncased)
```shell
export TASK_NAME=cola

accelerate launch examples/hf_transformers/general_language_understanding.py \
  --config configs/sample/glue/${TASK_NAME}/ce/bert_base_uncased.yaml \
  --task ${TASK_NAME} \
  -disable_cudnn_benchmark
```

### Write out test predictions for GLUE leaderboard submission
Since the test labels in GLUE are publicly available, you'll need to submit the test predictions to 
their leaderboard system to confirm the actual test performance.
With the proper `private` entries in your config file and `--private_output ${TEST_DIR}`, 
you can get the prediction file. Take a look at sample configs `configs/sample/glue/` for the entry format.  

e.g., Write out test predictions after fine-tuning BERT-Base (uncased)
```shell
export TASK_NAME=cola
export TEST_DIR=submission/standard/bert_base_uncased/

accelerate launch examples/hf_transformers/general_language_understanding.py \
  --config configs/sample/glue/${TASK_NAME}/ce/bert_base_uncased.yaml \
  --task ${TASK_NAME} \
  --private_output ${TEST_DIR} \
  -disable_cudnn_benchmark
```
Note that you should re-configure `accelerate` by `accelerate config` and avoid the distributed processing mode 
when writing out the prediction file. If you have a fine-tuned model, you can skip training by adding `-test_only`.

### Sample benchmark results and fine-tuned models
Using [this Google Colab example](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/dev/demo/glue_finetuning_and_submission.ipynb) 
and sample configs `configs/sample/glue/`, you should be able to get similar results.

#### BERT-Large (uncased) fine-tuned on each task
- Overall GLUE score: [80.2](https://gluebenchmark.com/leaderboard)
- Fine-tuned models: `yoshitomo-matsubara/bert-large-uncased-{dataset}` available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara)

#### BERT-Base (uncased) fine-tuned on each task
- Overall GLUE score: [77.9](https://gluebenchmark.com/leaderboard)
- Fine-tuned models: `yoshitomo-matsubara/bert-base-uncased-{dataset}` available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara)

#### BERT-Base (uncased) fine-tuned on each task, using fine-tuned BERT-Large (uncased) as teacher for knowledge distillation
- Overall GLUE score: [78.9](https://gluebenchmark.com/leaderboard)
- Fine-tuned models: `yoshitomo-matsubara/bert-base-uncased-{dataset}_from_bert-large-uncased-{dataset}` available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara)

## 3. Causal language modeling tasks (LMs)
[causal_language_modeling.py](causal_language_modeling.py) trains decoder-only LMs with or without a teacher.
The sample configs `configs/sample/alpaca/` use [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
(train split only, re-split into 90% train / 5% validation / 5% test) and cover three settings
with SmolLM2-135M-Instruct as the student:

| Setting | Config directory | Teacher |
| :-- | :-- | :-- |
| Instruction tuning (no teacher) | `configs/sample/alpaca/instruction_tuning/` | — |
| Off-policy KD | `configs/sample/alpaca/off_policy_kd/` | Alpaca-tuned SmolLM2-360M / 1.7B-Instruct |
| On-policy KD | `configs/sample/alpaca/on_policy_kd/` | Alpaca-tuned SmolLM2-360M / 1.7B-Instruct |

The metric is per-token perplexity over the completion tokens only (prompt tokens are masked out of the loss).
The best checkpoint is selected by validation perplexity and evaluated on the test split at the end.

### Instruction tuning
e.g., Fine-tuning SmolLM2-135M-Instruct on Alpaca (the teacher-free baseline for the KD configs below)
```shell
accelerate launch examples/hf_transformers/causal_language_modeling.py \
  --config configs/sample/alpaca/instruction_tuning/smollm2_135m.yaml \
  --run_log log/alpaca/instruction_tuning/smollm2_135m.txt \
  --seed 42
```

The teachers used in the KD configs, `yoshitomo-matsubara/smollm2_360m-alpaca-it` and
`yoshitomo-matsubara/smollm2_1_7b-alpaca-it`, are available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara)
and downloaded automatically. To reproduce them, run the same command with
`configs/sample/alpaca/instruction_tuning/smollm2_{360m,1_7b}.yaml`.

### Off-policy knowledge distillation
Both the teacher and the student are scored on the reference completions in the dataset, and
`OffPolicyKDLoss` minimizes the token-level KL divergence between them, combined with the student's task loss.

e.g., Distilling Alpaca-tuned SmolLM2-360M-Instruct into SmolLM2-135M-Instruct
```shell
accelerate launch examples/hf_transformers/causal_language_modeling.py \
  --config configs/sample/alpaca/off_policy_kd/smollm2_135m_from_smollm2_360m.yaml \
  --run_log log/alpaca/off_policy_kd/smollm2_135m_from_smollm2_360m.txt \
  --seed 42
```

### On-policy knowledge distillation
Before each forward pass, the student generates completions from the prompts (`generate_on_policy_sequences`),
and the teacher and the student are both scored on those self-generated sequences (`OnPolicyKDLoss`), following
[GKD (Agarwal et al., ICLR 2024)](https://openreview.net/forum?id=3zKtaqxLhW).
Apart from the source of the scored sequences, the on-policy configs match their off-policy counterparts, but
the generation step makes them much slower.

e.g., Distilling Alpaca-tuned SmolLM2-360M-Instruct into SmolLM2-135M-Instruct
```shell
accelerate launch examples/hf_transformers/causal_language_modeling.py \
  --config configs/sample/alpaca/on_policy_kd/smollm2_135m_from_smollm2_360m.yaml \
  --run_log log/alpaca/on_policy_kd/smollm2_135m_from_smollm2_360m.txt \
  --seed 42
```

For the larger teacher, use `smollm2_135m_from_smollm2_1_7b.yaml` in either directory.
Add `--mixed_precision fp16` (T4) or `--mixed_precision bf16` (Ampere or newer GPUs) to `accelerate launch`
for mixed-precision training.
Add `-test_only` to evaluate a trained checkpoint without training, and `-disable_tracker` to run without the
trackio tracker configured in the sample configs.

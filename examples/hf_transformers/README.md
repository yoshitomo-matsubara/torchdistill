# Examples of Use of Hugging Face Transformers

## 0. Requirements
### Install additional packages
You will first need to install Hugging Face's `transformers`, `datasets`, and `accelerate` packages.
Install dependencies using [requirements.txt](requirements.txt).  
```
pip3 install -r examples/hf_transformers/requirements.txt

# If you use pipenv, use the following instead
pipenv install -r examples/hf_transformers/requirements.txt
```

Unless you use your own (local) datasets, datasets will be automatically downloaded by `datasets` package 
when running the following scripts.

### Configure accelerate
Accelerate provides an easy API for multi-GPUs/TPU/fp16 (mix-precision).  
```
accelerate config

# If you use pipenv, use the following instead
pipenv run accelerate config
```

## 1. GLUE: Text classification
GLUE consists of 9 different tasks: CoLA, SST-2, MRPC, STS-B (regression), QQP, MNLI, QNLI, RTE, and WNLI.  
For these tasks, you can run experiments with `cola`, `sst2`, `mrpc`, `stsb`, `qqp`, `mnli`, `qnli`, `rte`, and `wnli`, respectively.

### Fine-tuning a Transformer model
e.g., Fine-tuning BERT-Base (uncased)
```
export TASK_NAME=cola

accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/samples/glue/${TASK_NAME}/ce/bert_base_uncased.yaml \
  --task ${TASK_NAME}
  
# use `pipenv run accelerate launch ...` if you're using pipenv
```

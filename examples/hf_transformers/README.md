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
Note that STS-B (TASK_NAME=`stsb`) is a regression task, and the sample config file is under `mse/` instead of `ce/`.

### Fine-tuning a Transformer model
e.g., Fine-tuning BERT-Base (uncased)
```
export TASK_NAME=cola

accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/sample/glue/${TASK_NAME}/ce/bert_base_uncased.yaml \
  --task ${TASK_NAME}
  
# use `pipenv run accelerate launch ...` if you're using pipenv
```

### Write out test predictions for GLUE leaderboard submission
Since the test labels in GLUE are publicly available, you'll need to submit the test predictions to 
their leaderboard system to confirm the actual test performance.
With the proper `private` entries in your config file and `--private_output ${TEST_DIR}`, 
you can get the prediction file. Take a look at sample configs `configs/sample/glue/` for the entry format.  

e.g., Write out test predictions after fine-tuning BERT-Base (uncased)
```
export TASK_NAME=cola
export TEST_DIR=submission/ce/bert_base_uncased/

accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/sample/glue/${TASK_NAME}/ce/bert_base_uncased.yaml \
  --task ${TASK_NAME} \
  --private_output ${TEST_DIR}
  
# use `pipenv run accelerate launch ...` if you're using pipenv
```
Note that you should re-configure `accelerate` by `accelerate config` and avoid the distributed processing mode 
when writing out the prediction file. If you have a fine-tuned model, you can skip training by adding `-test_only`.

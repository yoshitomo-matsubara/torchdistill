# torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP
## Citation
[[OpenReview](https://openreview.net/forum?id=A5Axeeu1Bo)] [[Preprint](https://arxiv.org/abs/2310.17644)]  
```bibtex
@article{matsubara2023torchdistill,
  title={{torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP}},
  author={Matsubara, Yoshitomo},
  journal={arXiv preprint arXiv:2310.17644},
  year={2023}
}
```

## Configuration
### Reported Results
#### GLUE Benchmark (test)
|                                       | MNLI-(m/mm) |  QQP | QNLI | SST-2 |    CoLA |     STS-B | MRPC |  RTE | WNLI |
|---------------------------------------|------------:|-----:|-----:|------:|--------:|----------:|-----:|-----:|-----:|
| Model (Method, Reference)             |   Acc./Acc. |   F1 | Acc. |  Acc. | M Corr. | P-S Corr. |   F1 | Acc. | Acc. |
|                                       |             |      |      |       |         |           |      |      |      |
| BERT-Large (FT, Devlin et al. (2019)) |   86.7/85.9 | 72.1 | 92.7 |  94.9 |    60.5 |      86.5 | 89.3 | 70.1 |  N/A |
| BERT-Large (FT, Ours)                 |   86.4/85.7 | 72.2 | 92.4 |  94.6 |    61.5 |      85.0 | 89.2 | 68.9 | 65.1 |
|                                       |             |      |      |       |         |           |      |      |      |
| BERT-Base (FT, Devlin et al. (2019))  |   84.6/83.4 | 71.2 | 90.5 |  93.5 |    52.1 |      85.8 | 88.9 | 66.4 |  N/A |
| BERT-Base (FT, Ours)                  |   84.2/83.3 | 71.4 | 91.0 |  94.1 |    51.1 |      84.4 | 86.8 | 66.7 | 65.8 |
| BERT-Base (KD, Ours)                  |   85.9/84.7 | 72.8 | 90.7 |  93.7 |    57.0 |      85.6 | 87.5 | 66.7 | 65.1 |

---
### Command to test with a checkpoint

Replace `key` of `student_model` entry with a model key you find at https://huggingface.co/yoshitomo-matsubara and 
run the script with `-test_only`

e.g., To test BERT-Base fine-tuned for CoLA task,
1. replace 'bert-base-uncased' at `key` of `student_model` entry in `configs/official/glue/yoshitomo-matsubara/nlp-oss2023/cola/ce/bert-base-uncased.yaml` with `yoshitomo-matsubara/bert-base-uncased-cola`
2. run `text_classification.py` with `-test_only`

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/cola/ce/bert_base_uncased.yaml \
  --task cola \
  --run_log log/glue/cola/ce/bert_base_uncased-test_only.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/ \
  -test_only
```


### GLUE Benchmark Submission

Check the following Google Colab examples:
- Fine-tuning without teacher models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/glue_finetuning_and_submission.ipynb)
- Knowledge distillation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/glue_kd_and_submission.ipynb)

### Fine-tuning BERT-Base

Configure Hugging Face Accelerate
```shell
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): 0
# How many processes in total will you use? [1]: 1
# Do you wish to use FP16 (mixed precision)? [yes/NO]: yes
```

#### Fine-tuning BERT-Base for CoLA task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/cola/ce/bert_base_uncased.yaml \
  --task cola \
  --run_log log/glue/cola/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for SST-2 task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/sst2/ce/bert_base_uncased.yaml \
  --task sst2 \
  --run_log log/glue/sst2/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for MRPC task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mrpc/ce/bert_base_uncased.yaml \
  --task mrpc \
  --run_log log/glue/mrpc/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for STS-B task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/stsb/mse/bert_base_uncased.yaml \
  --task stsb \
  --run_log log/glue/stsb/mse/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for QQP task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qqp/ce/bert_base_uncased.yaml \
  --task qqp \
  --run_log log/glue/qqp/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for MNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mnli/ce/bert_base_uncased.yaml \
  --task mnli \
  --run_log log/glue/mnli/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
     
```

#### Fine-tuning BERT-Base for QNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qnli/ce/bert_base_uncased.yaml \
  --task qnli \
  --run_log log/glue/qnli/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for RTE task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/rte/ce/bert_base_uncased.yaml \
  --task rte \
  --run_log log/glue/rte/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

#### Fine-tuning BERT-Base for WNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/wnli/ce/bert_base_uncased.yaml \
  --task wnli \
  --run_log log/glue/wnli/ce/bert_base_uncased.txt \
  --private_output leaderboard/glue/standard/bert_base_uncased/
```

### Fine-tuning BERT-Large

Configure Hugging Face Accelerate
```shell
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): 0
# How many processes in total will you use? [1]: 1
# Do you wish to use FP16 (mixed precision)? [yes/NO]: yes
```

#### Fine-tuning BERT-Large for CoLA task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/cola/ce/bert_large_uncased.yaml \
  --task cola \
  --run_log log/glue/cola/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for SST-2 task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/sst2/ce/bert_large_uncased.yaml \
  --task sst2 \
  --run_log log/glue/sst2/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for MRPC task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mrpc/ce/bert_large_uncased.yaml \
  --task mrpc \
  --run_log log/glue/mrpc/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for STS-B task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/stsb/mse/bert_large_uncased.yaml \
  --task stsb \
  --run_log log/glue/stsb/mse/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for QQP task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qqp/ce/bert_large_uncased.yaml \
  --task qqp \
  --run_log log/glue/qqp/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for MNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mnli/ce/bert_large_uncased.yaml \
  --task mnli \
  --run_log log/glue/mnli/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
     
```

#### Fine-tuning BERT-Large for QNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qnli/ce/bert_large_uncased.yaml \
  --task qnli \
  --run_log log/glue/qnli/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for RTE task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/rte/ce/bert_large_uncased.yaml \
  --task rte \
  --run_log log/glue/rte/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

#### Fine-tuning BERT-Large for WNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/wnli/ce/bert_large_uncased.yaml \
  --task wnli \
  --run_log log/glue/wnli/ce/bert_large_uncased.txt \
  --private_output leaderboard/glue/standard/bert_large_uncased/
```

### Distilling knowledge of BERT-Large into BERT-Base

Configure Hugging Face Accelerate
```shell
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): 0
# How many processes in total will you use? [1]: 1
# Do you wish to use FP16 (mixed precision)? [yes/NO]: yes
```

#### Distilling knowledge of BERT-Large into BERT-Base for CoLA task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/cola/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task cola \
  --run_log log/glue/cola/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for SST-2 task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/sst2/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task sst2 \
  --run_log log/glue/sst2/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for MRPC task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mrpc/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task mrpc \
  --run_log log/glue/mrpc/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for STS-B task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/stsb/mse/bert_base_uncased.yaml \
  --task stsb \
  --run_log log/glue/stsb/mse/bert_base_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for QQP task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qqp/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task qqp \
  --run_log log/glue/qqp/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for MNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/mnli/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task mnli \
  --run_log log/glue/mnli/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
     
```

#### Distilling knowledge of BERT-Large into BERT-Base for QNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/qnli/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task qnli \
  --run_log log/glue/qnli/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for RTE task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/rte/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task rte \
  --run_log log/glue/rte/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

#### Distilling knowledge of BERT-Large into BERT-Base for WNLI task

```shell
accelerate launch examples/hf_transformers/text_classification.py \
  --config configs/official/glue/yoshitomo-matsubara/nlp-oss2023/wnli/kd/bert_base_uncased_from_bert_large_uncased.yaml \
  --task wnli \
  --run_log log/glue/wnli/kd/bert_base_uncased_from_bert_large_uncased.txt \
  --private_output leaderboard/glue/kd/bert_base_uncased_from_bert_large_uncased/
```

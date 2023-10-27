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
#### CIFAR-100 (test)
| Model                         | Accuracy [%] |
|-------------------------------|-------------:|
| WRN-40-4                      |        79.44 |
| WRN-28-10                     |        81.27 |
| WRN-16-8                      |        79.26 |
| DenseNet-BC (k=12, depth=100) |        77.14 |

---
### Command to test pretrained models

Replace `pretrained: False` value of `model` entry with `pretrained: True` and run the script with `-test_only`  
e.g., WRN-40-4
```shell
python3 examples/torchvision/image_classification.py \
    --config configs/official/cifar100/yoshitomo-matsubara/nlp-oss2023/wide_resnet40_4-final_run.yaml \
    -test_only
```

### Command to train models

Google Colab example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/cifar_training.ipynb)

#### WRN-40-4

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar100/yoshitomo-matsubara/nlp-oss2023/wide_resnet40_4-final_run.yaml \
  --run_log log/cifar100/ce/wide_resnet40_4-final_run.log
```

#### WRN-28-10

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar100/yoshitomo-matsubara/nlp-oss2023/wide_resnet28_10-final_run.yaml \
  --run_log log/cifar100/ce/wide_resnet28_10-final_run.log
```

#### WRN-16-8

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar100/yoshitomo-matsubara/nlp-oss2023/wide_resnet16_8-final_run.yaml \
  --run_log log/cifar100/ce/wide_resnet16_8-final_run.log
```

#### DenseNet-BC (k=12, depth=100)

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar100/yoshitomo-matsubara/nlp-oss2023/densenet_bc_k12_depth100-final_run.yaml \
  --run_log log/cifar100/ce/densenet_bc_k12_depth100-final_run.log
```

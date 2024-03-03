# torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP
## Citation
[[OpenReview](https://openreview.net/forum?id=A5Axeeu1Bo)] [[Preprint](https://arxiv.org/abs/2310.17644)]  
```bibtex
@inproceedings{matsubara2023torchdistill,
  title={{torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP}},
  author={Matsubara, Yoshitomo},
  booktitle={Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)},
  publisher={Empirical Methods in Natural Language Processing},
  pages={153--164},
  year={2023}
}
```

## Configuration
### Reported Results
#### CIFAR-10 (test)
|                               |  Accuracy [%] |
|-------------------------------|--------------:|
| ResNet-20                     |         91.92 |
| ResNet-32                     |         93.03 |
| ResNet-44                     |         93.20 |
| ResNet-56                     |         93.57 |
| ResNet-110                    |         93.50 |
| WRN-40-4                      |         95.24 |
| WRN-28-10                     |         95.53 |
| WRN-16-8                      |         94.76 |
| DenseNet-BC (k=12, depth=100) |         95.53 |

---
### Command to test pretrained models

Replace `pretrained: False` value of `model` entry with `pretrained: True` and run the script with `-test_only`  
e.g., ResNet-20
```shell
python3 examples/torchvision/image_classification.py \
    --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet20-final_run.yaml \
    -test_only
```

### Command to train models

Google Colab example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/cifar_training.ipynb)

#### ResNet-20

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet20-final_run.yaml \
  --run_log log/cifar10/ce/resnet20-final_run.log
```

#### ResNet-32

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet32-final_run.yaml \
  --run_log log/cifar10/ce/resnet32-final_run.log
```

#### ResNet-44

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet44-final_run.yaml \
  --run_log log/cifar10/ce/resnet44-final_run.log
```

#### ResNet-56

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet56-final_run.yaml \
  --run_log log/cifar10/ce/resnet56-final_run.log
```

#### ResNet-110

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/resnet110-final_run.yaml \
  --run_log log/cifar10/ce/resnet110-final_run.log
```

#### WRN-40-4

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/wide_resnet40_4-final_run.yaml \
  --run_log log/cifar10/ce/wide_resnet40_4-final_run.log
```

#### WRN-28-10

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/wide_resnet28_10-final_run.yaml \
  --run_log log/cifar10/ce/wide_resnet28_10-final_run.log
```

#### WRN-16-8

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/wide_resnet16_8-final_run.yaml \
  --run_log log/cifar10/ce/wide_resnet16_8-final_run.log
```

#### DenseNet-BC (k=12, depth=100)

```shell
python3 examples/torchvision/image_classification.py \
  --config configs/official/cifar10/yoshitomo-matsubara/nlp-oss2023/densenet_bc_k12_depth100-final_run.yaml \
  --run_log log/cifar10/ce/densenet_bc_k12_depth100-final_run.log
```

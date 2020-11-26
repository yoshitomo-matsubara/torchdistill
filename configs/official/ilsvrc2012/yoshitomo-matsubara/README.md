# 1. torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
## Citation
```bibtex
@article{matsubara2020torchdistill,
  title={torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation},
  author={Matsubara, Yoshitomo},
  year={2020}
  eprint={2011.12913},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Configuration
### Models
- Teacher: ResNet-34
- Student: ResNet-18

### Reported Results
#### Top-1 validation accuracy of student model on ILSVRC 2012
| T: ResNet-34    | Pretrained | KD    | AT    | FT     | CRD   | Tf-KD | SSKD  | L2    | PAD-L2    |  
| :---            | ---:       | ---:  | ---:  | ---:   | ---:  | ---:  | ---:  | ---:  | ---:      |  
| S: ResNet-18\*  | 69.76      | 71.37 | 70.90 | 70.45  | 70.93 | 70.52 | 70.09 | 71.08 | 71.71     |  

### Checkpoints
[imagenet.zip](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.0.1/imagenet.zip)

### Command
#### Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/kd-resnet18_from_resnet34.yaml -test_only
```

#### Attention Transfer
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/at-resnet18_from_resnet34.yaml -test_only
```

#### Factor Transfer
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/ft-resnet18_from_resnet34.yaml -test_only
```

#### Teacher-free Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/tfkd-resnet18_from_resnet34.yaml -test_only
```

#### Semi-supervisioned Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/sskd-resnet18_from_resnet34.yaml -test_only
```

#### L2 (CSE + L2)
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/cse_l2-resnet18_from_resnet34.yaml -test_only
```

#### PAD-L2 (2nd stage)
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/pad_l2-resnet18_from_resnet34.yaml -test_only
```

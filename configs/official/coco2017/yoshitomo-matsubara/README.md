# 1. torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
## Citation
```bibtex
@article{matsubara2020kdkit,
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
- Teacher: Faster R-CNN with ResNet-50 and FPN (Pretrained)
- Student: Faster R-CNN with Bottleneck-injected ResNet-50 and FPN

### Reported Results

### Checkpoints

### Command
#### Generalized Head Network Distillation
```
python3 examples/object_detection.py --config configs/official/coco2017/ghnd-custom_fasterrcnn_resnet50_fpn_from_fasterrcnn_resnet50_fpn.yaml -test_only
```

### Models
- Teacher: Mask R-CNN with ResNet-50 and FPN
- Student: Mask R-CNN with Bottleneck-injected ResNet-50 and FPN

### Command
#### Generalized Head Network Distillation
```
python3 examples/object_detection.py --config configs/official/coco2017/ghnd-custom_maskrcnn_resnet50_fpn_from_maskrcnn_resnet50_fpn.yaml -test_only
```

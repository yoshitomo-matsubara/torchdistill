# torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
## Citation
[[Preprint](https://arxiv.org/abs/2011.12913)]  
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
### Reported Results
#### Validation mAP (0.5:0.95) of student model on COCO 2017
| T: Faster R-CNN with ResNet-50 and FPN                      | GHND (BBox)  |  
| :---                                                        | ---:         |  
| S: Bottleneck-injected Faster R-CNN with ResNet-50 and FPN  | 0.359        |  

| T: Mask R-CNN with ResNet-50 and FPN                      | GHND (BBox)  | GHND (Mask)  | 
| :---                                                      | ---:         | ---:         | 
| S: Bottleneck-injected Mask R-CNN with ResNet-50 and FPN  | 0.369        | 0.336        |  

### Checkpoints
[coco.zip](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.0.1/coco.zip)

---
### Command to test with checkpoints
- Download [coco.zip](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.0.1/coco.zip)
- Unzip `coco.zip` at the root directory of this repository

#### Generalized Head Network Distillation
- Teacher: Faster R-CNN with ResNet-50 and FPN (Pretrained)
- Student: Faster R-CNN with Bottleneck-injected ResNet-50 and FPN
```
python3 examples/object_detection.py --config configs/official/coco2017/yoshitomo-matsubara/rrpr2020/ghnd-custom_fasterrcnn_resnet50_fpn_from_fasterrcnn_resnet50_fpn.yaml -test_only
```

- Teacher: Mask R-CNN with ResNet-50 and FPN
- Student: Mask R-CNN with Bottleneck-injected ResNet-50 and FPN
```
python3 examples/object_detection.py --config configs/official/coco2017/yoshitomo-matsubara/rrpr2020/ghnd-custom_maskrcnn_resnet50_fpn_from_maskrcnn_resnet50_fpn.yaml -test_only
```

---
### Command for distributed training on 3 GPUs
1. Make sure checkpoint files do not exist at `ckpt` in `student_model` entry to train models from scratch.
2. Execute `export NUM_GPUS=3`

#### Generalized Head Network Distillation
- Teacher: Faster R-CNN with ResNet-50 and FPN (Pretrained)
- Student: Faster R-CNN with Bottleneck-injected ResNet-50 and FPN
```
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env examples/object_detection.py \
    --config configs/official/coco2017/yoshitomo-matsubara/rrpr2020/ghnd-custom_fasterrcnn_resnet50_fpn_from_fasterrcnn_resnet50_fpn.yaml \
    --log log/ghnd-custom_fasterrcnn_resnet50_fpn_from_fasterrcnn_resnet50_fpn.log \
    --world_size ${NUM_GPUS} 
```

- Teacher: Mask R-CNN with ResNet-50 and FPN
- Student: Mask R-CNN with Bottleneck-injected ResNet-50 and FPN
```
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env examples/object_detection.py \
    --config configs/official/coco2017/yoshitomo-matsubara/rrpr2020/ghnd-custom_maskrcnn_resnet50_fpn_from_maskrcnn_resnet50_fpn.yaml \
    --log log/coco2017/ghnd-custom_maskrcnn_resnet50_fpn_from_maskrcnn_resnet50_fpn.log \
    --world_size ${NUM_GPUS} 
```

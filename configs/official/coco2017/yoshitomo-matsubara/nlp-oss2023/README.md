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
#### Validation mIoU and global pixelwise accuracy of student model on COCO 2017
| T: DeepLabv3 with ResNet-50      | mIoU |  Pixelwise Acc. |  
|:---------------------------------|-----:|----------------:|  
| S: LRASPP with MobileNetV3-Large | 58.2 |            92.1 |  

---
### Command to test with a checkpoint
- Download https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.2.6/coco2017-lraspp_mobilenet_v3_large_from_deeplabv3_resnet50.pt
- Update `dst_ckpt` of student models defined in the yaml files in this directory with the checkpoint file path

#### Knowledge Translation and Adaptation + Affinity Distillation
- Teacher: DeepLabv3 with ResNet-50 (Pretrained)
- Student: LRASPP with MobileNetV3-Large

```
python3 examples/torchvision/semantic_segmentation.py \
    --config configs/official/coco2017/yoshitomo-matsubara/nlp-oss2023/ktaad-lraspp_mobilenet_v3_large_from_deeplabv3_resnet50.yaml \
    -test_only
```

---
### Command for distributed training on 3 GPUs
1. Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.
2. Execute `export NUM_GPUS=3`

#### Knowledge Translation and Adaptation + Affinity Distillation
- Teacher: DeepLabv3 with ResNet-50 (Pretrained)
- Student: LRASPP with MobileNetV3-Large

```
torchrun --nproc_per_node=${NUM_GPUS} examples/torchvision/semantic_segmentation.py \
    --config configs/official/coco2017/yoshitomo-matsubara/nlp-oss2023/ktaad-lraspp_mobilenet_v3_large_from_deeplabv3_resnet50.yaml \
    --run_log log/ktaad-lraspp_mobilenet_v3_large_from_deeplabv3_resnet50.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

# torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP
## Citation
[[OpenReview](https://openreview.net/forum?id=A5Axeeu1Bo)]
```bibtex
@inproceedings{matsubara2023torchdistill,
  title={{torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP}},
  author={Matsubara, Yoshitomo},
  booktitle={Proceedings of Third Workshop for NLP Open Source Software (NLP-OSS)},
  pages={},
  year={2023}
}
```

## Configuration
The configuration and log files in this directory are used for distributed training on 3 GPUs.  

### Models
- Teacher: ResNet-34
- Student: ResNet-18

### Reported Results
#### Top-1 validation accuracy of student model on ILSVRC 2012
| T: ResNet-34    | Pretrained |    KR |  
| :---            | ---:       |------:|  
| S: ResNet-18\*  | 69.76      | 71.64 |  


---
### Command to test with checkpoints
- Download https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.2.5/ilsvrc2012-resnet18_from_resnet34.pt
- Update `dst_ckpt` of student models defined in the yaml file in this directory with the checkpoint file path

#### Knowledge Review
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/nlp-oss2023/kr-resnet18_from_resnet34.yaml \
    -test_only
```


---
### Command for distributed training on 3 GPUs
1. Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.
2. Execute `export NUM_GPUS=3`

#### Knowledge Review
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/nlp-oss2023/kr-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/kr-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

# Understanding the Role of the Projector in Knowledge Distillation
## Citation
[[Preprint](https://arxiv.org/abs/2303.11098)]  
```bibtex
@inproceedings{miles2023understanding,
      title      = {Understanding the Role of the Projector in Knowledge Distillation}, 
      author     = {Roy Miles and Krystian Mikolajczyk},
      booktitle  = {Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)},
      year       = {2023},
      month      = {December}
}
```

## Configuration
The configuration and log files in this directory are used for distributed training on 1 GPU.  

### Models
- Teacher: ResNet-34
- Student: ResNet-18

### Reported Results
#### Top-1 validation accuracy of student model on ILSVRC 2012
| T: ResNet-34    | Pretrained |    SRD |  
| :---            | ---:       |------:|  
| S: ResNet-18\*  | 69.76      | 71.65 |  


---
### Command to test with checkpoints
- Download `.pt` checkpoint at: https://drive.google.com/drive/folders/1P5mePA0vwWkGqzJCiExfVzpqZEpEDEEz?usp=sharing
- Update `dst_ckpt` of student models defined in the yaml file in this directory with the checkpoint file path

#### Simple Recipe Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/roymiles/aaai2024/srd-resnet18_from_resnet34.yaml \
    -test_only
```


---
### Command for distributed training on 3 GPUs
1. Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.
2. Execute `export NUM_GPUS=1`

#### Simple Recipe Distillation
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/roymiles/aaai2024/srd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/srd-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

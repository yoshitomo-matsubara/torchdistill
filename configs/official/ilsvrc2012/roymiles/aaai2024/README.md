# Understanding the Role of the Projector in Knowledge Distillation
## Citation
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28219)] [[Preprint](https://arxiv.org/abs/2303.11098)]  
```bibtex
@inproceedings{miles2024understanding,
  title={Understanding the Role of the Projector in Knowledge Distillation},
  author={Miles, Roy and Mikolajczyk, Krystian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4233--4241},
  year={2024}
}
```

## Configuration
The configuration and log files in this directory are used for distributed training on 1 GPU.  

### Models
- Teacher: ResNet-34
- Student: ResNet-18

### Reported Results
#### Top-1 validation accuracy of student model on ILSVRC 2012
| T: ResNet-34    | Pretrained |   SRD |  
| :---            | ---:       |------:|  
| S: ResNet-18\*  | 69.76      | 71.65 |  


---
### Command to test with checkpoints
- Download https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v1.1.0/ilsvrc2012.zip from [Google Drive]( https://drive.google.com/drive/folders/1P5mePA0vwWkGqzJCiExfVzpqZEpEDEEz?usp=sharing)
- Update `src_ckpt` of student models defined in the yaml file in this directory with the checkpoint file path

#### Simple Recipe Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/roymiles/aaai2024/srd-resnet18_from_resnet34.yaml \
    -test_only
```


---
### Command for training without distributed processes
Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.

#### Simple Recipe Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/roymiles/aaai2024/srd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/srd-resnet18_from_resnet34.log 
```

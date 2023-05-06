# torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
## Citation
[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3)] [[Preprint](https://arxiv.org/abs/2011.12913)]  
```bibtex
@inproceedings{matsubara2021torchdistill,
  title={torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation},
  author={Matsubara, Yoshitomo},
  booktitle={International Workshop on Reproducible Research in Pattern Recognition},
  pages={24--44},
  year={2021},
  organization={Springer}
}
```

## Configuration
The configuration and log files in this directory are used for either 1) distributed training on 3 GPUs or 
2) training on 1 GPU that resulted in better accuracy.  

You can find configuration and log files for both 1) and 2) in `imagenet.zip`.

### Models
- Teacher: ResNet-34
- Student: ResNet-18

### Reported Results
#### Top-1 validation accuracy of student model on ILSVRC 2012
| T: ResNet-34    | Pretrained | KD    | AT    | FT     | CRD   | Tf-KD | SSKD  | L2    | PAD-L2    |  
| :---            | ---:       | ---:  | ---:  | ---:   | ---:  | ---:  | ---:  | ---:  | ---:      |  
| S: ResNet-18\*  | 69.76      | 71.37 | 70.90 | 71.56  | 70.93 | 70.52 | 70.09 | 71.08 | 71.71     |  

### Checkpoints
[imagenet.zip](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.0.1/imagenet.zip)

---
### Command to test with checkpoints
- Download [imagenet.zip](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.0.1/imagenet.zip)
- Unzip `imagenet.zip`
- Update `dst_ckpt` of student models defined in the yaml files in this directory with unzipped checkpoint file paths

#### Knowledge Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/kd-resnet18_from_resnet34.yaml \
    -test_only
```

#### Attention Transfer
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/at-resnet18_from_resnet34.yaml \
    -test_only
```

#### Contrastive Representation Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/crd-resnet18_from_resnet34.yaml \
    -test_only
```

#### Factor Transfer
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/ft-resnet18_from_resnet34.yaml \
    -test_only
```

#### Teacher-free Knowledge Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/tfkd-resnet18_from_resnet18.yaml \
    -test_only
```

#### Semi-supervisioned Knowledge Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/sskd-resnet18_from_resnet34.yaml \
    -test_only
```

#### L2 (CSE + L2)
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/cse_l2-resnet18_from_resnet34.yaml \
    -test_only
```

#### PAD-L2 (2nd stage)
Note that you first need to train a model with L2 (CSE + L2), and load the ckpt file designated in the following yaml file.  
i.e., PAD-L2 is a two-stage training method.

```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/pad_l2-resnet18_from_resnet34.yaml \
    -test_only
```

---
### Command for distributed training on 3 GPUs
1. Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.
2. Execute `export NUM_GPUS=3`

#### Knowledge Distillation
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/kd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/kd-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

#### Attention Transfer
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/at-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/at-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

#### Factor Transfer
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/ft-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/ft-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

#### Contrastive Representation Distillation
If you use fewer or more GPUs for distributed training, you should update `batch_size: 85` in `train_data_loader` entry 
so that (batch size) * ${NUM_GPUS}  = 256. (e.g., `batch_size: 32` if you use 8 GPUs for distributed training.)  

```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/crd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/crd-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} 
```

#### Teacher-free Knowledge Distillation
```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/tfkd-resnet18_from_resnet18.yaml \
    --run_log log/ilsvrc2012/tfkd-resnet18_from_resnet18.log \
    --world_size ${NUM_GPUS} \
    -adjust_lr
```

#### Semi-supervisioned Knowledge Distillation
If you use fewer or more GPUs for distributed training, you should update `batch_size: 85` in `train_data_loader` entry 
so that (batch size) * ${NUM_GPUS}  = 256. (e.g., `batch_size: 32` if you use 8 GPUs for distributed training.)  

```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/sskd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/sskd-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} 
```

#### L2 (CSE + L2)
If you use fewer or more GPUs for distributed training, you should update `batch_size: 171` in `train_data_loader` entry 
so that (batch size) * ${NUM_GPUS}  = 512. (e.g., `batch_size: 64` if you use 8 GPUs for distributed training.)  

```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/cse_l2-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/cse_l2-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} 
```

#### PAD-L2 (2nd stage)
Note that you first need to train a model with L2 (CSE + L2), and load the ckpt file designated in the following yaml file.  
i.e., PAD-L2 is a two-stage training method.  

If you use fewer or more GPUs for distributed training, you should update `batch_size: 171` in `train_data_loader` entry 
so that (batch size) * ${NUM_GPUS}  = 512. (e.g., `batch_size: 64` if you use 8 GPUs for distributed training.) 

```
torchrun  --nproc_per_node=${NUM_GPUS} examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/pad_l2-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/pad_l2-resnet18_from_resnet34.log \
    --world_size ${NUM_GPUS} 
```
Multi-stage methods can be defined in one yaml file like [this](https://github.com/yoshitomo-matsubara/torchdistill/blob/master/configs/sample/image_classification/multi_stage/pad), 
but you should modify the hyperparameters like number of epochs, lr scheduler, and so on.

---
### Command for training without distributed processes
Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.

#### Knowledge Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/kd-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/kd-resnet18_from_resnet34.log
```

#### Attention Transfer
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/at-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/at-resnet18_from_resnet34.log
```

#### Factor Transfer
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/ft-resnet18_from_resnet34.yaml \
    --run_log log/ilsvrc2012/ft-resnet18_from_resnet34.log
```

#### Teacher-free Knowledge Distillation
```
python3 examples/torchvision/image_classification.py \
    --config configs/official/ilsvrc2012/yoshitomo-matsubara/rrpr2020/tfkd-resnet18_from_resnet18.yaml \
    --run_log log/ilsvrc2012/tfkd-resnet18_from_resnet18.log
```

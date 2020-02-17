# Knowledge distillation kit for PyTorch

## Requirements
- Python 3.6
- pipenv
- [myutils](https://github.com/yoshitomo-matsubara/myutils)


## How to setup
```
git clone https://github.com/yoshitomo-matsubara/kdkit.git
cd kdkit/
git submodule init
git submodule update --recursive --remote
pipenv install
```

## Examples
### 1. ImageNet
#### 1.1 Download and unzip ImageNet datasets (ILSVRC2012 in this example)
#### 1.2 Execute the following commands
```
cd kdkit/
mkdir ./resource/dataset/ilsvrc2012/{train,val} -p

# Download the training and validation datasets from ImageNet website
# Untar them under train and val dirs respectively

wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valpre.sh ./resource/dataset/ilsvrc2012/val/
cd ./resource/dataset/ilsvrc2012/val/
sh valpre.sh
```
#### 1.3 Distill knowledge of ResNet-152
e.g., Teacher: ResNet-152, Student: AlexNet  
a) Use GPU(s) for single training process
```
pipenv run python src/image_classification.py --config config/image_classification/kd/alexnet_from_resnet152.yaml
```  
b) Use GPUs for multiple distributed training processes
```
pipenv run python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env src/image_classification.py --world_size ${NUM_GPUS} --config config/image_classification/kd/alexnet_from_resnet152.yaml
```
c) Use CPU
```
pipenv run python src/image_classification.py --device cpu --config config/image_classification/kd/alexnet_from_resnet152.yaml
```  
#### 1.4 Top 1 accuracy of student models
| Teacher \\ Student    | AlexNet   | ResNet-18 |  
| :---                  | ---:      | ---:      |  
| Pretrained (no *KD*)  | 56.52     | 69.76     |  
| ResNet-152            | 57.22     | 69.86     |


## References
- [pytorch/vision/references/classification/](https://github.com/pytorch/vision/blob/master/references/classification/)

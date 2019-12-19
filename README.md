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
1. Download and unzip ImageNet datasets (ILSVRC2012 in this example)
2. Execute the following commands
```
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
```

## References
- [pytorch/vision/references/classification/](https://github.com/pytorch/vision/blob/master/references/classification/)

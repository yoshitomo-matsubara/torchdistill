# torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP
## Citation
[[Paper](https://aclanthology.org/2023.nlposs-1.18/)] [[OpenReview](https://openreview.net/forum?id=A5Axeeu1Bo)] [[Preprint](https://arxiv.org/abs/2310.17644)]  
```bibtex
@inproceedings{matsubara2023torchdistill,
  title={{torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP}},
  author={Matsubara, Yoshitomo},
  booktitle={Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)},
  publisher={Empirical Methods in Natural Language Processing},
  pages={153--164},
  year={2023}
}
```

## Configuration
### Reported Results
#### Validation mIoU and global pixelwise accuracy of student model on PASCAL VOC 2012 (Segmentation)
| Model                     | mIoU |    Pixelwise Acc. |  
|:--------------------------|-----:|------------------:|  
| DeepLabv3 with ResNet-50  | 80.6 |              95.7 |  
| DeepLabv3 with ResNet-101 | 82.4 |              96.2 |  

---
### Command to test with a checkpoint
- Download [pascal_voc2012-deeplabv3_resnet50.pt](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.2.8/pascal_voc2012-deeplabv3_resnet50.pt) and [pascal_voc2012-deeplabv3_resnet101.pt](https://github.com/yoshitomo-matsubara/torchdistill/releases/download/v0.2.8/pascal_voc2012-deeplabv3_resnet101.pt)
- Update `src_ckpt` of student models defined in the yaml files in this directory with the checkpoint file path

DeepLabv3 with ResNet-50
```
python3 examples/torchvision/semantic_segmentation.py \
    --config configs/official/pascal_voc2012/yoshitomo-matsubara/nlp-oss2023/deeplabv3_resnet50.yaml \
    -test_only
```

DeepLabv3 with ResNet-101
```
python3 examples/torchvision/semantic_segmentation.py \
    --config configs/official/pascal_voc2012/yoshitomo-matsubara/nlp-oss2023/deeplabv3_resnet101.yaml \
    -test_only
```

---
### Command for training without distributed processes
Make sure checkpoint files do not exist at `dst_ckpt` in `student_model` entry to train models from scratch.

DeepLabv3 with ResNet-50
```
python3 examples/torchvision/semantic_segmentation.py \
    --config configs/official/pascal_voc2012/yoshitomo-matsubara/nlp-oss2023/deeplabv3_resnet50.yaml \
    --run_log log/deeplabv3_resnet50.log 
```

DeepLabv3 with ResNet-101
```
python3 examples/torchvision/semantic_segmentation.py \
    --config configs/official/pascal_voc2012/yoshitomo-matsubara/nlp-oss2023/deeplabv3_resnet101.yaml \
    --run_log log/deeplabv3_resnet101.log 
```

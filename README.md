# torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
[![PyPI version](https://badge.fury.io/py/torchdistill.svg)](https://badge.fury.io/py/torchdistill)
[![Build Status](https://travis-ci.com/yoshitomo-matsubara/torchdistill.svg?branch=master)](https://travis-ci.com/github/yoshitomo-matsubara/torchdistill)
[![DOI:10.1007/978-3-030-76423-4_3](https://zenodo.org/badge/DOI/10.1007/978-3-030-76423-4_3.svg)](https://doi.org/10.1007/978-3-030-76423-4_3)


***torchdistill*** (formerly *kdkit*) offers various knowledge distillation methods 
and enables you to design (new) experiments simply by editing a yaml config file instead of Python code. 
Even when you need to extract intermediate representations in teacher/student models, 
you will **NOT** need to reimplement the models, that often change the interface of the forward, but instead 
specify the module path(s) in the yaml file. Refer to [this paper](https://github.com/yoshitomo-matsubara/torchdistill#citation) for more details.  

In addition to knowledge distillation, this framework enables you to train models without teachers simply by 
excluding teacher entries from a yaml config file. You can find such examples below and in [configs/sample/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/).

## Forward hook manager
Using **ForwardHookManager**, you can extract intermediate representations in model without modifying the interface of its forward function.  
[This example notebook](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/demo/extract_intermediate_representations.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/extract_intermediate_representations.ipynb) 
will give you a better idea of the usage such as knowledge distillation and analysis of intermediate representations.

## 1 experiment → 1 PyYAML config file
In ***torchdistill***, many components and PyTorch modules are abstracted e.g., models, datasets, optimizers, losses, 
and more! You can define them in a PyYAML config file so that can be seen as a summary of your experiment, and 
in many cases, you will **NOT need to write Python code at all**. 
Take a look at some configurations available in [configs/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/). You'll see what modules are abstracted and 
how they are defined in a PyYAML config file to design an experiment.

## Top-1 validation accuracy for ILSVRC 2012 (ImageNet)
| T: ResNet-34\*  | Pretrained | KD    | AT    | FT         | CRD   | Tf-KD | SSKD  | L2    | PAD-L2    | Knowledge Review |  
| :---            | ---:       | ---:  | ---:  | ---:       | ---:  | ---:  | ---:  | ---:  | ---:      | ---:             |  
| S: ResNet-18    | 69.76\*    | 71.37 | 70.90 | 71.56      | 70.93 | 70.52 | 70.09 | 71.08 | 71.71     | 71.64            |  
| Original work   | N/A        | N/A   | 70.70 | 71.43\*\*  | 71.17 | 70.42 | 71.62 | 70.90 | 71.71     | 71.61            |  
  
\* The pretrained ResNet-34 and ResNet-18 are provided by torchvision.  
\*\* FT is assessed with ILSVRC 2015 in the original work.  
For the 2nd row (S: ResNet-18), most of the results are reported in [this paper](https://github.com/yoshitomo-matsubara/torchdistill#citation), 
and their checkpoints (trained weights), configuration and log files are [available](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/official/ilsvrc2012/yoshitomo-matsubara/), 
and the configurations reuse the hyperparameters such as number of epochs used in the original work except for KD.

## Examples
Executable code can be found in [examples/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/) such as
- [Image classification](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/image_classification.py): ImageNet (ILSVRC 2012), CIFAR-10, CIFAR-100, etc
- [Object detection](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/object_detection.py): COCO 2017, etc
- [Semantic segmentation](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/semantic_segmentation.py): COCO 2017, PASCAL VOC, etc
- [Text classification](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/hf_transformers/text_classification.py): GLUE, etc

For CIFAR-10 and CIFAR-100, some models are reimplemented and available as pretrained models in ***torchdistill***. 
More details can be found [here](https://github.com/yoshitomo-matsubara/torchdistill/releases/tag/v0.1.1).  

Some Transformer models fine-tuned by ***torchdistill*** for GLUE tasks are available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara). 
Sample GLUE benchmark results and details can be found [here](https://github.com/yoshitomo-matsubara/torchdistill/tree/master/examples/hf_transformers#sample-benchmark-results-and-fine-tuned-models).

## Google Colab Examples
The following examples are available in [demo/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/demo/). 
Note that the examples are for Google Colab users. Usually, [examples/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/) would be a better reference 
if you have your own GPU(s).

### CIFAR-10 and CIFAR-100
- Training without teacher models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/cifar_training.ipynb)
- Knowledge distillation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/cifar_kd.ipynb)

### GLUE
- Fine-tuning without teacher models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/glue_finetuning_and_submission.ipynb)
- Knowledge distillation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/master/demo/glue_kd_and_submission.ipynb)

These examples write out test prediction files for you to see the test performance at [the GLUE leaderboard system](https://gluebenchmark.com/).

## PyTorch Hub
If you find models on [PyTorch Hub](https://pytorch.org/hub/) or GitHub repositories supporting PyTorch Hub,
you can import them as teacher/student models simply by editing a yaml config file.  

e.g., If you use a pretrained ResNeSt-50 available in [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
as a teacher model for ImageNet dataset, you can import the model via PyTorch Hub with the following entry in your yaml
config file.

```yaml
models:
  teacher_model:
    name: 'resnest50d'
    repo_or_dir: 'rwightman/pytorch-image-models'
    params:
      num_classes: 1000
      pretrained: True
```

## How to setup
- Python 3.6 >=
- pipenv (optional)

### Install by pip/pipenv
```
pip3 install torchdistill
# or use pipenv
pipenv install torchdistill
```

### Install from this repository 
```
git clone https://github.com/yoshitomo-matsubara/torchdistill.git
cd torchdistill/
pip3 install -e .
# or use pipenv
pipenv install "-e ."
```

## Issues / Questions / Requests
The documentation is work-in-progress. In the meantime, feel free to create an issue if you find a bug.  
If you have either a question or feature request, start a new discussion [here](https://github.com/yoshitomo-matsubara/torchdistill/discussions).

## Citation
If you use ***torchdistill*** in your research, please cite the following paper.  
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

## References
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/image_classification.py) [pytorch/vision/references/classification/](https://github.com/pytorch/vision/blob/master/references/classification/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/object_detection.py) [pytorch/vision/references/detection/](https://github.com/pytorch/vision/tree/master/references/detection/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/semantic_segmentation.py) [pytorch/vision/references/segmentation/](https://github.com/pytorch/vision/tree/master/references/segmentation/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/hf_transformers/text_classification.py) [huggingface/transformers/examples/pytorch/text-classification](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/kd) Geoffrey Hinton, Oriol Vinyals and Jeff Dean. ["Distilling the Knowledge in a Neural Network"](https://fb56552f-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningworkshopnips2014/65.pdf?attachauth=ANoY7co8sQACDsEYLkP11zqEAxPgYHLwkdkDP9NHfEB6pzQOUPmfWf3cVrL3WE7PNyed-lrRsF7CY6Tcme5OEQ92CTSN4f8nDfJcgt71fPtAvcTvH5BpzF-2xPvLkPAvU9Ub8XvbySAPOsMKMWmGsXG2FS1_X1LJsUfuwKdQKYVVTtRfG5LHovLHIwv6kXd3mOkDKEH7YdoyYQqjSv6ku2KDjOpVQBt0lKGVPXeRdwUcD0mxDqCe4u8%3D&attredirects=1) (Deep Learning and Representation Learning Workshop: NeurIPS 2014)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/fitnet) Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta and Yoshua Bengio. ["FitNets: Hints for Thin Deep Nets"](https://arxiv.org/abs/1412.6550) (ICLR 2015)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/fsp) Junho Yim, Donggyu Joo, Jihoon Bae and Junmo Kim. ["A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html) (CVPR 2017)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/at) Sergey Zagoruyko and Nikos Komodakis. ["Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer"](https://openreview.net/forum?id=Sks9_ajex) (ICLR 2017)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/pkt) Nikolaos Passalis and Anastasios Tefas. ["Learning Deep Representations with Probabilistic Knowledge Transfer"](http://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html) (ECCV 2018)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/ft) Jangho Kim, Seonguk Park and Nojun Kwak. ["Paraphrasing Complex Network: Network Compression via Factor Transfer"](http://papers.neurips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer) (NeurIPS 2018)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/dab) Byeongho Heo, Minsik Lee, Sangdoo Yun and Jin Young Choi. ["Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"](https://aaai.org/ojs/index.php/AAAI/article/view/4264) (AAAI 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/rkd) Wonpyo Park, Dongju Kim, Yan Lu and Minsu Cho. ["Relational Knowledge Distillation"](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) (CVPR 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/vid) Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence and Zhenwen Dai. ["Variational Information Distillation for Knowledge Transfer"](http://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html) (CVPR 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/hnd) Yoshitomo Matsubara, Sabur Baidya, Davide Callegaro, Marco Levorato and Sameer Singh. ["Distilled Split Deep Neural Networks for Edge-Assisted Real-Time Systems"](https://dl.acm.org/doi/10.1145/3349614.3356022) (Workshop on Hot Topics in Video Analytics and Intelligent Edges: MobiCom 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/cckd) Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou and Zhaoning Zhang. ["Correlation Congruence for Knowledge Distillation"](http://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) (ICCV 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/spkd) Frederick Tung and Greg Mori. ["Similarity-Preserving Knowledge Distillation"](http://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html) (ICCV 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/crd) Yonglong Tian, Dilip Krishnan and Phillip Isola. ["Contrastive Representation Distillation"](https://openreview.net/forum?id=SkgpBJrtvS) (ICLR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/coco2017/single_stage/ghnd) Yoshitomo Matsubara and Marco Levorato. ["Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"](https://arxiv.org/abs/2007.15818) (ICPR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/tfkd) Li Yuan, Francis E.H.Tay, Guilin Li, Tao Wang and Jiashi Feng. ["Revisiting Knowledge Distillation via Label Smoothing Regularization"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf) (CVPR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/sskd) Guodong Xu, Ziwei Liu, Xiaoxiao Li and Chen Change Loy. ["Knowledge Distillation Meets Self-Supervision"](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php) (ECCV 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/multi_stage/pad) Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang and Yichen Wei. ["Prime-Aware Adaptive Distillation"](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php) (ECCV 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/single_stage/kr) Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia. ["Distilling Knowledge via Knowledge Review"](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html) (CVPR 2021)

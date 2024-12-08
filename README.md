![torchdistill logo](https://raw.githubusercontent.com/yoshitomo-matsubara/torchdistill/main/docs/source/_static/images/logo-color.png)

# torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
[![PyPI version](https://img.shields.io/pypi/v/torchdistill?color=00cc00)](https://pypi.org/project/torchdistill/)
[![Build Status](https://app.travis-ci.com/yoshitomo-matsubara/torchdistill.svg?token=BsLDELKSQ5j23aFsTQKc&branch=main)](https://travis-ci.com/github/yoshitomo-matsubara/torchdistill)
[![GitHub Discussions](https://img.shields.io/github/discussions/yoshitomo-matsubara/torchdistill)](https://github.com/yoshitomo-matsubara/torchdistill/discussions)
[![DOI:10.1007/978-3-030-76423-4_3](https://zenodo.org/badge/DOI/10.1007/978-3-030-76423-4_3.svg)](https://doi.org/10.1007/978-3-030-76423-4_3)
[![DOI:10.18653/v1/2023.nlposs-1.18](https://zenodo.org/badge/DOI/10.18653/v1/2023.nlposs-1.18.svg)](https://doi.org/10.18653/v1/2023.nlposs-1.18)


***torchdistill*** (formerly *kdkit*) offers various state-of-the-art knowledge distillation methods 
and enables you to design (new) experiments simply by editing a declarative yaml config file instead of Python code. 
Even when you need to extract intermediate representations in teacher/student models, 
you will **NOT** need to reimplement the models, that often change the interface of the forward, but instead 
specify the module path(s) in the yaml file. Refer to [these papers](https://github.com/yoshitomo-matsubara/torchdistill#citation) for more details.  

In addition to knowledge distillation, this framework helps you design and perform general deep learning experiments
(**WITHOUT coding**) for reproducible deep learning studies. i.e., it enables you to train models without teachers 
simply by excluding teacher entries from a declarative yaml config file. 
You can find such examples below and in [configs/sample/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/).   

When you refer to ***torchdistill*** in your paper, please cite [these papers](https://github.com/yoshitomo-matsubara/torchdistill#citation) 
instead of this GitHub repository.  
**If you use** ***torchdistill*** **as part of your work, your citation is appreciated and motivates me to maintain and upgrade this framework!** 


## Documentation
You can find the API documentation and research projects that leverage ***torchdistill*** at https://yoshitomo-matsubara.net/torchdistill/


## Forward hook manager
Using **ForwardHookManager**, you can extract intermediate representations in model without modifying the interface of its forward function.  
[This example notebook](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/demo/extract_intermediate_representations.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/main/demo/extract_intermediate_representations.ipynb) [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/yoshitomo-matsubara/torchdistill/blob/main/demo/extract_intermediate_representations.ipynb) 
will give you a better idea of the usage such as knowledge distillation and analysis of intermediate representations.


## 1 experiment → 1 declarative PyYAML config file
In ***torchdistill***, many components and PyTorch modules are abstracted e.g., models, datasets, optimizers, losses, 
and more! You can define them in a declarative PyYAML config file so that can be seen as a summary of your experiment, 
and in many cases, you will **NOT need to write Python code at all**. 
Take a look at some configurations available in [configs/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/). 
You'll see what modules are abstracted and how they are defined in a declarative PyYAML config file to design an experiment.  

If you want to use your own modules (models, loss functions, datasets, etc) with this framework, 
you can do so without editing code in the local package `torchdistill/`.  
See [the official documentation](https://yoshitomo-matsubara.net/torchdistill/usage.html) and [Discussions](https://github.com/yoshitomo-matsubara/torchdistill/discussions) for more details. 

## Benchmarks

[Top-1 validation accuracy for ILSVRC 2012 (ImageNet)](https://yoshitomo-matsubara.net/torchdistill/benchmarks.html#imagenet-ilsvrc-2012)


## Examples
Executable code can be found in [examples/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/) such as
- [Image classification](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/torchvision/image_classification.py): ImageNet (ILSVRC 2012), CIFAR-10, CIFAR-100, etc
- [Object detection](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/torchvision/object_detection.py): COCO 2017, etc
- [Semantic segmentation](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/torchvision/semantic_segmentation.py): COCO 2017, PASCAL VOC, etc
- [Text classification](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/hf_transformers/text_classification.py): GLUE, etc

For CIFAR-10 and CIFAR-100, some models are reimplemented and available as pretrained models in ***torchdistill***. 
More details can be found [here](https://github.com/yoshitomo-matsubara/torchdistill/releases/tag/v0.1.1).  

Some Transformer models fine-tuned by ***torchdistill*** for GLUE tasks are available at [Hugging Face Model Hub](https://huggingface.co/yoshitomo-matsubara). 
Sample GLUE benchmark results and details can be found [here](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/hf_transformers#sample-benchmark-results-and-fine-tuned-models).


## Google Colab Examples
The following examples are available in [demo/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/demo/). 
Note that these examples are for Google Colab users and compatible with Amazon SageMaker Studio Lab. 
Usually, [examples/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/) would be a better reference 
if you have your own GPU(s).

### CIFAR-10 and CIFAR-100
- Training without teacher models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/main/demo/cifar_training.ipynb) [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/yoshitomo-matsubara/torchdistill/blob/main/demo/cifar_training.ipynb)
- Knowledge distillation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/main/demo/cifar_kd.ipynb) [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/yoshitomo-matsubara/torchdistill/blob/main/demo/cifar_kd.ipynb)

### GLUE
- Fine-tuning without teacher models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/main/demo/glue_finetuning_and_submission.ipynb) [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/yoshitomo-matsubara/torchdistill/blob/main/demo/glue_finetuning_and_submission.ipynb)
- Knowledge distillation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshitomo-matsubara/torchdistill/blob/main/demo/glue_kd_and_submission.ipynb) [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/yoshitomo-matsubara/torchdistill/blob/main/demo/glue_kd_and_submission.ipynb)

These examples write out test prediction files for you to see the test performance at [the GLUE leaderboard system](https://gluebenchmark.com/).


## PyTorch Hub
If you find models on [PyTorch Hub](https://pytorch.org/hub/) or GitHub repositories supporting PyTorch Hub,
you can import them as teacher/student models simply by editing a declarative yaml config file.  

e.g., If you use a pretrained ResNeSt-50 available in [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
(aka *timm*) as a teacher model for ImageNet dataset, you can import the model via PyTorch Hub with the following entry 
in your declarative yaml config file.

```yaml
models:
  teacher_model:
    key: 'resnest50d'
    repo_or_dir: 'huggingface/pytorch-image-models'
    kwargs:
      num_classes: 1000
      pretrained: True
```


## How to setup
- Python >= 3.9
- pipenv (optional)

### Install by pip/pipenv
```
pip3 install torchdistill
# or use pipenv
pipenv install torchdistill
```

### Install from this repository (not recommended)
```
git clone https://github.com/yoshitomo-matsubara/torchdistill.git
cd torchdistill/
pip3 install -e .
# or use pipenv
pipenv install "-e ."
```


## Issues / Questions / Requests / Pull Requests
Feel free to create an issue if you find a bug.  
If you have either a question or feature request, start a new discussion [here](https://github.com/yoshitomo-matsubara/torchdistill/discussions).
Please search through [Issues](https://github.com/yoshitomo-matsubara/torchdistill/issues) and [Discussions](https://github.com/yoshitomo-matsubara/torchdistill/discussions) and make sure your issue/question/request has not been addressed yet.

Pull requests are welcome. 
Please start with an issue and discuss solutions with me rather than start with a pull request. 

## Citation
If you use ***torchdistill*** in your research, please cite the following papers:  
[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3)] [[Preprint](https://arxiv.org/abs/2011.12913)]  
```bibtex
@inproceedings{matsubara2021torchdistill,
  title={{torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation}},
  author={Matsubara, Yoshitomo},
  booktitle={International Workshop on Reproducible Research in Pattern Recognition},
  pages={24--44},
  year={2021},
  organization={Springer}
}
```

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


## Acknowledgments

This project has been supported by Travis CI's OSS credits and [JetBrain's Free License Programs (Open Source)](https://www.jetbrains.com/community/opensource/?utm_campaign=opensource&utm_content=approved&utm_medium=email&utm_source=newsletter&utm_term=jblogo#support) 
since November 2021 and June 2022, respectively.  
![PyCharm logo](https://resources.jetbrains.com/storage/products/company/brand/logos/PyCharm.svg)


## References
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/image_classification.py) [pytorch/vision/references/classification/](https://github.com/pytorch/vision/blob/main/references/classification/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/object_detection.py) [pytorch/vision/references/detection/](https://github.com/pytorch/vision/tree/main/references/detection/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/semantic_segmentation.py) [pytorch/vision/references/segmentation/](https://github.com/pytorch/vision/tree/main/references/segmentation/)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/hf_transformers/text_classification.py) [huggingface/transformers/examples/pytorch/text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/kd/) Geoffrey Hinton, Oriol Vinyals, Jeff Dean. ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) (Deep Learning and Representation Learning Workshop: NeurIPS 2014)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/fitnet/) Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio. ["FitNets: Hints for Thin Deep Nets"](https://arxiv.org/abs/1412.6550) (ICLR 2015)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/fsp/) Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim. ["A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html) (CVPR 2017)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/at/) Sergey Zagoruyko, Nikos Komodakis. ["Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer"](https://openreview.net/forum?id=Sks9_ajex) (ICLR 2017)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/pkt/) Nikolaos Passalis, Anastasios Tefas. ["Learning Deep Representations with Probabilistic Knowledge Transfer"](http://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html) (ECCV 2018)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/ft/) Jangho Kim, Seonguk Park, Nojun Kwak. ["Paraphrasing Complex Network: Network Compression via Factor Transfer"](http://papers.neurips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer) (NeurIPS 2018)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/dab/) Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi. ["Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"](https://aaai.org/ojs/index.php/AAAI/article/view/4264) (AAAI 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/coco2017/ktaad/) Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan. ["Knowledge Adaptation for Efficient Semantic Segmentation"](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html) (CVPR 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/rkd/) Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho. ["Relational Knowledge Distillation"](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) (CVPR 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/vid/) Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai. ["Variational Information Distillation for Knowledge Transfer"](http://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html) (CVPR 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/hnd/) Yoshitomo Matsubara, Sabur Baidya, Davide Callegaro, Marco Levorato, Sameer Singh. ["Distilled Split Deep Neural Networks for Edge-Assisted Real-Time Systems"](https://dl.acm.org/doi/10.1145/3349614.3356022) (Workshop on Hot Topics in Video Analytics and Intelligent Edges: MobiCom 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/cckd/) Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, Zhaoning Zhang. ["Correlation Congruence for Knowledge Distillation"](http://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) (ICCV 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/spkd/) Frederick Tung, Greg Mori. ["Similarity-Preserving Knowledge Distillation"](http://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html) (ICCV 2019)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/crd/) Yonglong Tian, Dilip Krishnan, Phillip Isola. ["Contrastive Representation Distillation"](https://openreview.net/forum?id=SkgpBJrtvS) (ICLR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/coco2017/ghnd/) Yoshitomo Matsubara, Marco Levorato. ["Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"](https://arxiv.org/abs/2007.15818) (ICPR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/tfkd/) Li Yuan, Francis E.H.Tay, Guilin Li, Tao Wang, Jiashi Feng. ["Revisiting Knowledge Distillation via Label Smoothing Regularization"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf) (CVPR 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/sskd/) Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy. ["Knowledge Distillation Meets Self-Supervision"](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php) (ECCV 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/pad/) Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang, Yichen Wei. ["Prime-Aware Adaptive Distillation"](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php) (ECCV 2020)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/ilsvrc2012/kr/) Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia. ["Distilling Knowledge via Knowledge Review"](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html) (CVPR 2021)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/blob/main/configs/sample/ilsvrc2012/ickd/) Li Liu, Qingle Huang, Sihao Lin, Hongwei Xie, Bing Wang, Xiaojun Chang, Xiaodan Liang. ["Exploring Inter-Channel Correlation for Diversity-Preserved Knowledge Distillation"](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html) (ICCV 2021)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/blob/main/configs/sample/ilsvrc2012/dist/) Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu. ["Knowledge Distillation from A Stronger Teacher"](https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html) (NeurIPS 2022)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/blob/main/configs/sample/ilsvrc2012/srd/) Roy Miles, Krystian Mikolajczyk. ["Understanding the Role of the Projector in Knowledge Distillation"](https://ojs.aaai.org/index.php/AAAI/article/view/28219) (AAAI 2024)
- [:mag:](https://github.com/yoshitomo-matsubara/torchdistill/blob/main/configs/sample/ilsvrc2012/kd_w_ls/) Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao. ["Logit Standardization in Knowledge Distillation"](https://arxiv.org/abs/2403.01427) (CVPR 2024)

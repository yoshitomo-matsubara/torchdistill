Projects
=====


.. toctree::
   :maxdepth: 2
   :caption: Overview

This page is a showcase of OSS (open source software) and papers which have used **torchdistill** in the projects.
If your work is built on **torchdistill**, start `a "Show and tell" discussion at GitHub <https://github.com/yoshitomo-matsubara/torchdistill/discussions/new?category=show-and-tell>`_.


OSS
*****

sc2bench
----
* PyPI: https://pypi.org/project/sc2bench/
* Code: https://github.com/yoshitomo-matsubara/sc2-benchmark

This framework was built on PyTorch and designed to benchmark SC2 methods, *Supervised Compression for Split Computing*.
It is pip-installable and published as a PyPI package i.e., you can install it by :code:`pip3 install sc2bench`


-----

Papers
*****

torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP
----
* Author(s): Yoshitomo Matsubara
* Venue: EMNLP 2023 Workshop for Natural Language Processing Open Source Software (NLP-OSS)
* PDF: `Paper <https://aclanthology.org/2023.nlposs-1.18/>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/torchdistill>`_

**Abstract**: Reproducibility in scientific work has been becoming increasingly important in research communities
such as machine learning, natural language processing, and computer vision communities due to the rapid development of
the research domains supported by recent advances in deep learning. In this work, we present a significantly upgraded
version of torchdistill, a modular-driven coding-free deep learning framework significantly upgraded from the initial
release, which supports only image classification and object detection tasks for reproducible knowledge distillation
experiments. To demonstrate that the upgraded framework can support more tasks with third-party libraries, we reproduce
the GLUE benchmark results of BERT models using a script based on the upgraded torchdistill, harmonizing with various
Hugging Face libraries. All the 27 fine-tuned BERT models and configurations to reproduce the results are published at
Hugging Face, and the model weights have already been widely used in research communities. We also reimplement popular
small-sized models and new knowledge distillation methods and perform additional experiments for computer vision tasks.


SC2 Benchmark: Supervised Compression for Split Computing
----
* Author(s): Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt
* Venue: TMLR
* PDF: `Paper + Supp <https://openreview.net/forum?id=p28wv4G65d>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/sc2-benchmark>`_

**Abstract**: With the increasing demand for deep learning models on mobile devices, splitting neural network
computation between the device and a more powerful edge server has become an attractive solution. However, existing
split computing approaches often underperform compared to a naive baseline of remote computation on compressed data.
Recent studies propose learning compressed representations that contain more relevant information for supervised
downstream tasks, showing improved tradeoffs between compressed data size and supervised performance. However, existing
evaluation metrics only provide an incomplete picture of split computing. This study introduces supervised compression
for split computing (SC2) and proposes new evaluation criteria: minimizing computation on the mobile device, minimizing
transmitted data size, and maximizing model accuracy. We conduct a comprehensive benchmark study using 10 baseline
methods, three computer vision tasks, and over 180 trained models, and discuss various aspects of SC2. We also release
our code and sc2bench, a Python package for future research on SC2. Our proposed metrics and package will help
researchers better understand the tradeoffs of supervised compression in split computing.


Supervised Compression for Resource-Constrained Edge Computing Systems
----
* Author(s): Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt
* Venue: WACV 2022
* PDF: `Paper + Supp <https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/supervised-compression>`_

**Abstract**: There has been much interest in deploying deep learning algorithms on low-powered devices, including
smartphones, drones, and medical sensors. However, full-scale deep neural networks are often too resource-intensive
in terms of energy and storage. As a result, the bulk part of the machine learning operation is therefore often
carried out on an edge server, where the data is compressed and transmitted. However, compressing data (such as images)
leads to transmitting information irrelevant to the supervised task. Another popular approach is to split the deep
network between the device and the server while compressing intermediate features. To date, however, such split
computing strategies have barely outperformed the aforementioned naive data compression baselines due to their
inefficient approaches to feature compression. This paper adopts ideas from knowledge distillation and neural image
compression to compress intermediate feature representations more efficiently. Our supervised compression approach
uses a teacher model and a student model with a stochastic bottleneck and learnable prior for entropy coding
(Entropic Student). We compare our approach to various neural image and feature compression baselines in three vision
tasks and found that it achieves better supervised rate-distortion performance while maintaining smaller end-to-end
latency. We furthermore show that the learned feature representations can be tuned to serve multiple downstream tasks.

torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation
----
* Author(s): Yoshitomo Matsubara
* Venue: ICPR 2020 International Workshop on Reproducible Research in Pattern Recognition
* PDF: `Paper <https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3>`_
* Code: `GitHub <https://github.com/yoshitomo-matsubara/torchdistill>`_

**Abstract**: While knowledge distillation (transfer) has been attracting attentions from the research community,
the recent development in the fields has heightened the need for reproducible studies and highly generalized frameworks
to lower barriers to such high-quality, reproducible deep learning research. Several researchers voluntarily published
frameworks used in their knowledge distillation studies to help other interested researchers reproduce their original
work. Such frameworks, however, are usually neither well generalized nor maintained, thus researchers are still
required to write a lot of code to refactor/build on the frameworks for introducing new methods, models, datasets and
designing experiments. In this paper, we present our developed open-source framework built on PyTorch and dedicated for
knowledge distillation studies. The framework is designed to enable users to design experiments by declarative PyYAML
configuration files, and helps researchers complete the recently proposed ML Code Completeness Checklist. Using the
developed framework, we demonstrate its various efficient training strategies, and implement a variety of knowledge
distillation methods. We also reproduce some of their original experimental results on the ImageNet and COCO datasets
presented at major machine learning conferences such as ICLR, NeurIPS, CVPR and ECCV, including recent state-of-the-art
methods. All the source code, configurations, log files and trained model weights are publicly available at
https://github.com/yoshitomo-matsubara/torchdistill.

Benchmarks
=====


.. toctree::
   :maxdepth: 2
   :caption: Overview

This page summarizes reproducible experimental results that `torchdistill official repository <https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/official>`_ supports.


ImageNet (ILSVRC 2012)
*****

`ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) <https://www.image-net.org/challenges/LSVRC/2012/>`_

----

Student: ResNet-18
----

.. csv-table:: Top-1 validation accuracy of ResNet-18 for ILSVRC 2012 (ImageNet)
   :file: _static/benchmarks/imagenet-resnet18_kd.tsv
   :delim: tab
   :align: center
   :header-rows: 1

\* The pretrained ResNet-34 and ResNet-18 are provided by torchvision.

\*\* FT is assessed with ILSVRC 2015 in the original work.

Original work
^^^^

* KD: `"Distilling the Knowledge in a Neural Network" <https://arxiv.org/abs/1503.02531>`_
* AT: `"Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer" <https://openreview.net/forum?id=Sks9_ajex>`_
* FT: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <http://papers.neurips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer>`_
* CRD: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_
* Tf-KD: `"Revisiting Knowledge Distillation via Label Smoothing Regularization" <https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf>`_
* SSDK: `"Knowledge Distillation Meets Self-Supervision" <http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_
* :math:`L_2`, PAD-:math:`L_2`: `"Prime-Aware Adaptive Distillation" <http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php>`_
* KR: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_


References
^^^^
* Yoshitomo Matsubara: `"torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation" <https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3>`_
* Yoshitomo Matsubara: `"torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP" <https://arxiv.org/abs/2310.17644>`_

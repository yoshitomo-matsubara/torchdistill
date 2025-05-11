torchdistill.models
=====


.. toctree::
   :maxdepth: 4
   :caption: Contents:


----

torchdistill.models.registry
------------

.. automodule:: torchdistill.models.registry
   :members:

----

torchdistill.models.classification
------------

----

To reproduce the test results for CIFAR datasets, the following repositories were referred for training methods:

- ResNet: https://github.com/facebookarchive/fb.resnet.torch
- WRN (Wide ResNet): https://github.com/szagoruyko/wide-residual-networks
- DenseNet-BC: https://github.com/liuzhuang13/DenseNet

.. list-table:: Accuracy of models pretrained on CIFAR-10/100 datasets
   :widths: 50 25 25
   :header-rows: 1

   * - Model
     - CIFAR-10
     - CIFAR-100
   * - `ResNet-20 <#torchdistill.models.classification.resnet.resnet20>`_
     - 91.92
     - N/A
   * - `ResNet-32 <#torchdistill.models.classification.resnet.resnet32>`_
     - 93.03
     - N/A
   * - `ResNet-44 <#torchdistill.models.classification.resnet.resnet44>`_
     - 93.20
     - N/A
   * - `ResNet-56 <#torchdistill.models.classification.resnet.resnet56>`_
     - 93.57
     - N/A
   * - `ResNet-110 <#torchdistill.models.classification.resnet.resnet110>`_
     - 93.50
     - N/A
   * - `WRN-40-4 <#torchdistill.models.classification.wide_resnet.wide_resnet40_4>`_
     - 95.24
     - 79.44
   * - `WRN-28-10 <#torchdistill.models.classification.wide_resnet.wide_resnet28_10>`_
     - 95.53
     - 81.27
   * - `WRN-16-8 <#torchdistill.models.classification.wide_resnet.wide_resnet16_8>`_
     - 94.76
     - 79.26
   * - `DenseNet-BC (k=12, depth=100) <#torchdistill.models.classification.densenet.densenet_bc_k12_depth100>`_
     - 95.53
     - 77.14

Those results are reported in the following paper:

* Yoshitomo Matsubara: `"torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP" <https://aclanthology.org/2023.nlposs-1.18/>`_

.. automodule:: torchdistill.models.classification
   :members:

----

torchdistill.models.classification.densenet
^^^^^^^^^^^^

.. automodule:: torchdistill.models.classification.densenet
   :members:
   :exclude-members: forward

----

torchdistill.models.classification.resnet
^^^^^^^^^^^^

.. automodule:: torchdistill.models.classification.resnet
   :members:
   :exclude-members: forward

----

torchdistill.models.classification.wide_resnet
^^^^^^^^^^^^

.. automodule:: torchdistill.models.classification.wide_resnet
   :members:
   :exclude-members: forward

----

torchdistill.models.official
------------

.. automodule:: torchdistill.models.official
   :members:
   :exclude-members: forward

----

torchdistill.models.adaptation
------------

.. automodule:: torchdistill.models.adaptation
   :members:
   :exclude-members: forward

----

torchdistill.models.wrapper
------------

.. automodule:: torchdistill.models.wrapper
   :members:
   :exclude-members: forward

----

torchdistill.models.util
------------

.. automodule:: torchdistill.models.util
   :members:
   :exclude-members: forward


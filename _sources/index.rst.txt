.. torchdistill documentation master file, created by
   sphinx-quickstart on Mon Aug 21 23:14:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/images/logo-color.png
  :alt: torchdistill logo

torchdistill documentation
========================================

**torchdistill** (formerly *kdkit*) offers various state-of-the-art knowledge distillation methods
and enables you to design (new) experiments simply by editing a declarative yaml config file instead of Python code.
Even when you need to extract intermediate representations in teacher/student models,
you will **NOT** need to reimplement the models, that often change the interface of the forward, but instead
specify the module path(s) in the yaml file.

In addition to knowledge distillation, this framework helps you design and perform general deep learning experiments
(**WITHOUT coding**) for reproducible deep learning studies. i.e., it enables you to train models without teachers
simply by excluding teacher entries from a declarative yaml config file.
You can find such examples in `configs/sample/ of the official repository <https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/>`_.

When you refer to **torchdistill** in your paper, please cite `these papers <https://github.com/yoshitomo-matsubara/torchdistill#citation>`_
instead of this GitHub repository.
**If you use torchdistill as part of your work, your citation is appreciated and motivates me to maintain and upgrade this framework!**

.. toctree::
   :maxdepth: 2
   :caption: 📚 Overview

   usage
   package

.. toctree::
   :maxdepth: 2
   :caption: 🧑🏻‍💻 Research

   benchmarks
   projects


References
*********
.. code-block:: bibtex

   @inproceedings{matsubara2021torchdistill,
     title={{torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation}},
     author={Matsubara, Yoshitomo},
     booktitle={International Workshop on Reproducible Research in Pattern Recognition},
     pages={24--44},
     year={2021},
     organization={Springer}
   }

   @inproceedings{matsubara2023torchdistill,
     title={{torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP}},
     author={Matsubara, Yoshitomo},
     booktitle={Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)},
     publisher={Empirical Methods in Natural Language Processing},
     pages={153--164},
     year={2023}
   }


Questions / Requests
==================

If you have either a question or feature request, start `a new "Q&A" discussion at GitHub <https://github.com/yoshitomo-matsubara/torchdistill/discussions>`_ instead of a GitHub issue.
Please make sure the issue/question/request has not been addressed yet by searching through the open/closed issues and discussions.


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

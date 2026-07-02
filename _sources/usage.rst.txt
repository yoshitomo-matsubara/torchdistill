Usage
=====


.. toctree::
   :maxdepth: 2
   :caption: Overview


Installation
------------

To use `torchdistill <https://pypi.org/project/torchdistill/>`_, first install it using pip:

.. code-block:: console

   $ pip install torchdistill


Examples
------------

`The official repository (https://github.com/yoshitomo-matsubara/torchdistill) <https://github.com/yoshitomo-matsubara/torchdistill>`_ offers many example scripts, configs,
and checkpoints of trained models in `torchdistill`.

Currently, `example scripts <https://github.com/yoshitomo-matsubara/torchdistill/tree/main/examples/>`_
cover the following tasks:

- Image classification (ILSVRC 2012, CIFAR-10/100)
- Object detection (COCO 2017)
- Semantic segmentation (PASCAL VOC 2012, COCO 2017)
- Text classification (GLUE tasks)


How to Add Your Modules
------------

Step 1: Define your own module

Step 2: Register the module e.g., add a registry function to the module as a Python decorator

Step 3: Run your script with a yaml file containing the module name (key) and parameters, call the Python decorator, and then your module is available in the registry

.. code-block:: python
   :caption: Steps 1 and 2: Create a Python file (e.g., *my_module.py*) containing your own module (e.g., "MyNewCoolModel") with a Python decorator "register_model"

    from torch import nn
    from torchdistill.models.registry import register_model

    @register_model
    class MyNewCoolModel(nn.Module):
        def __init__(self, some_value, some_list, some_dict):
            super().__init__()
            print('some_value: ', some_value)
            print('some_list: ', some_list)
            print('some_dict: ', some_dict)
        ...

.. code-block:: yaml
   :caption: Step 3: Run your script (e.g., *example/torchvision/image_classification.py*) with a yaml containing the registered module name ("MyNewCoolModel") and parameters ("some_value", "some_list", "some_dict")

    dependencies:
      - name: 'my_module'
    ...
    models:
      model:
        key: 'MyNewCoolModel'
        kwargs:
          some_value: 777
          some_list: ['this', 'is', 'some_list']
          some_dict:
            some_key: 'some_value'
            test: 0.123
        src_ckpt:
    ...

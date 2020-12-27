from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

description = 'A Modular, Configuration-Driven Framework for Knowledge Distillation. ' \
              'Trained models, training logs and configurations are available for ensuring the reproducibiliy.'
setup(
    name='torchdistill',
    version='0.1.0',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/torchdistill',
    packages=find_packages(exclude=('tests', 'examples', 'demo', 'configs')),
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'numpy',
        'pyyaml',
        'scipy',
        'cython',
        'pycocotools>=2.0.1'
    ],
    extras_require={
        'test': ['flake8', 'pytest']
    },
)

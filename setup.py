from setuptools import setup, find_packages

import torchdistill

with open('README.md', 'r') as f:
    long_description = f.read()

description = 'A Modular, Configuration-Driven Framework for Knowledge Distillation. ' \
              'Trained models, training logs and configurations are available for ensuring the reproducibility.'
setup(
    name='torchdistill',
    version=torchdistill.__version__,
    author='Yoshitomo Matsubara',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/torchdistill',
    packages=find_packages(exclude=('tests', 'examples', 'demo', 'docs', 'configs')),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.1',
        'numpy',
        'pyyaml>=6.0',
        'scipy',
        'cython'
    ],
    extras_require={
        'test': ['pytest'],
        'docs': ['sphinx', 'sphinx_rtd_theme', 'sphinx_sitemap']
    }
)

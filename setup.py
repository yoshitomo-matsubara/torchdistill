from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='torchdistill',
    version='0.0.1',
    description='A unified knowledge distillation framework.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/torchdistill',
    packages=find_packages(exclude=('tests', 'examples', 'config')),
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.6.0',
        'torchvision>=0.7',
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

from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'modalic>=0.1.0'
    'torch>=1.11.0'
    'torchvision>=0.12.0',
]


def setup_package():
    __version__ = '0.1.0'

    setup(name='pytorch_mnist',
          description='Pytorch Federated Learning example classifying Hand-written.',
          version=__version__,
          install_requires=requirements,
          packages=find_packages(),
          )


if __name__ == '__main__':
    setup_package()

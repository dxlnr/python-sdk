import os, sys
import numpy as np
from codecs import open
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'grpcio>=1.43.0',
    'grpcio-tools>=1.43.0',
    'minio>=1.0.1.1',
    'toml>=0.10.2',
    'black',
]

here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)

version = {}
with open(os.path.join(here, "modalic", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version)

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]

setup(name='modalic',
      description='Python SDK library for the using the modalic MLOps platform.',
      version=version['VERSION'],
      url='https://github.com/modalic/python-sdk',
      author='Modalic',
      license='MIT',
      install_requires=requirements,
      include_dirs=getInclude(),
      packages=find_packages(exclude=["tests", "tests.*"]),
      classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

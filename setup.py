import os, sys
from codecs import open
from setuptools import setup, Extension, find_packages

requirements = [
    'grpcio>=1.43.0',
    'grpcio-tools>=1.43.0',
    'minio>=1.0.1.1',
    'toml>=0.10.2',
]

here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)

version_contents = {}
with open(os.path.join(here, "modalic", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version_contents)

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

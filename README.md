<h1 align="center">
  <b>Modalic Python SDK</b><br>
</h1>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Python SDK library for the using the Modalic MLOps Federated Learning platform.

## Installation

### Binaries
Commands to install binaries via conda or pip wheels via
```bash
pip install modalic
```

### From Source

#### Prerequisites
Installing from source, the following prerequisites are needed:
- Python 3.8 or later
- [Rust](https://www.rust-lang.org/tools/install) Toolchain for compiling the server application.
- [Anaconda](https://www.anaconda.com/distribution/#download-section) environment is recommended.

#### Install Dependencies

```bash
# Running in some conda environment
conda install setuptools, setuptools-rust
```

#### Get the Modalic Python-SDK Source
```bash
git clone --recursive https://github.com/modalic/python-sdk
cd python-sdk

git submodule sync
git submodule update --init --recursive
```

#### Install Modalic Python-SDK
On **Linux**

```bash
# Building the wheel (.whl)
python setup.py sdist bdist_wheel
# Installing the wheel locally
pip install dist/modalic-0.1.0-cp39-cp39-linux_x86_64.whl
```

### Docker Image
Install docker:  https://docs.docker.com/engine/install/ubuntu/

#### Using pre-built images

There is the option of using a pre-built docker image and run  in interactive mode with docker v19.03+

```bash
docker run
```

#### Building the image yourself
```bash
./build_....
```

## Implementation Ideas
- Testing / Parser that checks if the framework architecture is correctly set up and implemented beforehand to avoid python giving run time error.
- Add the server as wrapped dockercontainer within Github.

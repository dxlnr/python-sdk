<h1 align="center">
  <b>Modalic Python SDK</b><br>
</h1>

  [![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/modalic/python-sdk/blob/main/CONTRIBUTING.md)

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
git submodule -q foreach git pull -q origin main
git pull --recurse-submodules
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

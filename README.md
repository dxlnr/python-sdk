![Modalic Logo](https://github.com/modalic/python-sdk/blob/main/docs/source/_static/mo-logo.png)
<!-- ![test](https://raw.githubusercontent.com/modalic/python-sdk/main/docs/source/_static/mo-logo.svg?token=GHSAT0AAAAAABRDIVC2OKVSTFHDANG5FISUYWVIGDA) -->

--------------------------------------------------------------------------------

<h1 align="center">
  <b>Python SDK</b><br>
</h1>

  [![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/modalic/python-sdk/blob/main/CONTRIBUTING.md)

Python SDK library for using the Modalic MLOps Federated Learning platform.
The SDK library serves as convenient interface for performing Federated Learning with the most common machine learning frameworks ([Pytorch](https://github.com/pytorch/pytorch)) written in the Python programming language. As an API layer it connects an individual client application into a Federated Learning infrastructre as it is compatible to a server application provided by Modalic.

## Usage

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

## Documentation

See the [Python SDK docs](https://docs.modalic.ai/) for more information. Additionally, some examples for starting with Modalic are provided in this repository.

## Development

Find more information on contributing and the development process [here](CONTRIBUTING.md).

## License

The Modalic Python SDK is distributed under the terms of the Apache License Version 2.0. A complete version of the license is available in [LICENSE](LICENSE).

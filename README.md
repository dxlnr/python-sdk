![Modalic Logo](https://github.com/modalic/python-sdk/blob/main/docs/source/_static/mo-logo.png)

--------------------------------------------------------------------------------

<h1 align="center">
  <b>Python SDK</b><br>
</h1>

<p align="center">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-2F54D1.svg" /></a>
    <a href="https://github.com/modalic/python-sdk/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-apache2-351c75.svg" /></a>
    <a href="https://github.com/modalic/python-sdk/blob/main/CONTRIBUTING.md">
      <img src="https://img.shields.io/badge/PRs-welcome-6834D5.svg" /></a>
</p>

Python SDK library for using the **Modalic Federated Learning Operations Platform**.
The SDK library serves as convenient interface for performing Federated Learning with the most common Machine Learning frameworks
like [Pytorch](https://github.com/pytorch/pytorch) written in the Python programming language.
As an API layer it enables an individual client application to take part within a Federated Learning setup.
The coordination of a distributed Machine Learning process solving a particular problem,
is done by a central server or service provider which can be started via Python script using the SDK.

As the main entrypoint to a production ready FLOps Platform, this software package aims for all developers and ML practitioners that want to run ML use cases in distributed fashion.

## Usage
In order to run a Federated Learning procedure two main entities have to instantiated. The client logic and the aggregation server application.
Both can be started via SDK. This uses Pytorch as framework to construct the ML architecture.

```python
# (1) run the aggregation server with configuration using .toml (cfg)
modalic.run_server(cfg)

  # .toml
  #
  # [api]
  # server_address = "[::]:8080"
  #
  # [model]
  # data_type = "F32"
  #
  # [process]
  # training_rounds = 10
  # participants = 3
  # strategy = "FedAvg"

# (2) Construct the client logic.

# Define a Trainer object that contains all the ML logic.
class Trainer():

  def __init__():
    self.model = Net()
    self.dataset = torch.utils.data.DataLoader(dataset, batch_size=32)
    ...

  def train():
    ...

# Put the Modalic client layer on top of the ML logic.
client = modalic.PytorchClient(Trainer())

# (3) Run training for single client.
client.run()
```

Please keep in mind that this code snippet shows only the logic and the general idea. For more details,
check out the */examples* folder that contains more in-depth and complete instruction sets and examples that are actually actionable.

## Installation

### Binaries
The latest release of Modalic Python SDK can be installed via pip:
```bash
pip install modalic
```

### From Source

#### Prerequisites
Installing from source, the following prerequisites are needed:
- Python 3.8 or later
- [Rust](https://www.rust-lang.org/tools/install) Toolchain for compiling the server application.
- [Anaconda](https://www.anaconda.com/distribution/#download-section) environment is recommended. But any other virtual environment should be fine as well.

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
pip install dist/modalic-*.whl
```

## Documentation

See the [Python SDK docs](https://docs.modalic.ai/) for more information. Additionally, some examples for starting with Modalic are provided in this repository under the examples folder. Any Questions? Reach out to us on [modalic.ai](https://modalic.ai//contact).

## Development

Find more information on contributing to the open source stack and the development process in general [here](CONTRIBUTING.md).

## License

The Modalic Python SDK is distributed under the terms of the Apache License Version 2.0. A complete version of the license is available in [LICENSE](LICENSE).

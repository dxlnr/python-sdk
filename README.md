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

The SDK library serves as convenient interface for performing Federated Learning with the most common Machine Learning frameworks like [Pytorch](https://github.com/pytorch/pytorch) or [Tensorflow](https://github.com/tensorflow/tensorflow) written in the Python programming language.
As an additional software layer within a Machine Learning pipeline, the SKD enables an individual client application to take part within a Federated Learning setup.
The coordination of a distributed Machine Learning process solving a particular problem, is done by a central server or service provider which can be started via Python script using the SDK.

As the main entrypoint to a production ready FLOps Platform, this software package aims for all developers and ML practitioners that want to run ML use cases in distributed fashion.

## Usage
In order to run a Federated Learning procedure two main entities have to instantiated. The client logic and the aggregation server application.
Both can be started via the SDK. Currently Pytorch \& Tensorflow are supported as framework to construct the ML architecture.

#### Run the Aggregation Server

```python
# (1) Run the aggregation server with configuration using .toml
cfg = toml.load("${configPATH}.toml")
modalic.run_server(cfg)
```
The *.toml* file can be used to control hyperparameters for the aggregation server.
```toml
# -c configs/config.toml

# REST API settings.
[api]
# The address to which the REST API of the server
# will be bound. All requests should be sent to this address.
server_address = "127.0.0.1:8080"

# Hyperparameter controlling the Federated Learning training process.
[protocol]
# Defines the number of training rounds (global epochs) 
# that will be performed.
training_rounds = 10
# Sets the number of participants & local models 
# one global epoch should at least contain.
participants = 2
```

For implementing the client logic a framework of choice can be used.

#### Pytorch

```python
# (2) Construct the client logic.
import modalic

# Define a FLClient object that implements all the ML logic and will
# used as an input to an internal modalic client which enables the 
# program to connect to the server an perform training in distributed fashion.
class FLClient(modalic.Client):

  def __init__(self, dataset, ...):
    self.model = Net()
    self.dataset = torch.utils.data.DataLoader(dataset, batch_size=32)
    ...

  def train(self):
    for epoch in range(0, self.epochs):
        for images, labels in self.trainloader:
            ...

    return self.model

  def serialize_local_model(self, model):
      return modalic.serialize_torch_model(model)

  def deserialize_global_model(self, global_model):
      self.model = modalic.deserialize_torch_model(
          self.model, global_model, self._get_model_shape()
      )

  def get_model_shape(self):
      return modalic.get_torch_model_shape(self.model)

  def get_model_dtype(self):
      ...

# Construct the client layer..
client = FLClient(...)

# (3) Run training for single client.
modalic.run_client(client)
```

#### Tensorflow

```python
# (2) Construct the client logic.

# Wrap the custom defined train function with modalic.tf_train.
@modalic.tf_train(...)
def train(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=32, epochs=1)
    return model
    ...

# Define the model & data
model = tf.keras.Model(...)
...

# (3) Run training for single client simply via.
train(model, x_train, y_train)
```

Please keep in mind that this code snippet shows only the logic and the general idea. For more details, check out the */examples* folder that contains more in-depth and complete instruction sets and examples that are actually actionable.

## Installation

The latest release of Modalic Python SDK can be installed via pip:
```bash
pip install modalic
```

## Documentation

See the [Python SDK docs](https://docs.modalic.ai/) for more information. Additionally, some examples for starting with Modalic are provided in this repository under the examples folder. Any Questions? Reach out to us on [modalic.ai](https://modalic.ai//contact).

## Development

Find more information on contributing to the open source stack and the development process in general [here](CONTRIBUTING.md).

## License

The Modalic Python SDK is distributed under the terms of the Apache License Version 2.0. A complete version of the license is available in [LICENSE](LICENSE).

.. _getting-started:

Getting Started
===============

Installation
------------

For installing the Python SDK and further information regarding the process,
please visit the `installation guide <installation>`_.

It is recommended to create a virtual environment with `Ananconda <https://anaconda.org/>`_ and run everything within
the `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ setup.
Otherwise, installation is simply done by running::

  pip install modalic

Client Python SDK
-----------------

The :ref:`Python SDK <python-sdk-ref>` serves as an API endpoint and a general toolkit for Federated Learning on the client side.
It aims for a simple and quick integration within the Machine Learning stack, that defines the learning task.
Important to note, that the SDK does not imply what type of model for what kind of problem space has to be
defined but rather lets the developer control the stack entirely.


SDK Endpoints
-------------

There is basically one main endpoint that enable the client's ability to participate in a Federated Learning
procedure. Integrating FL into one's Machine Learning stack, is done by implementing the ML logic by using the :ref:`modalic.Client <modalic-client>`.

.. code-block:: python

  # Define a FLClient object that implements all the ML logic and will
  # used as an input to an internal modalic client which enables the 
  # program to connect to the server an perform training in distributed fashion.
  class FLClient(modalic.Client):

    def __init__(self, dataset, ...):
      self.model = Net()
      self.dataset = dataset
      ...

    def train(self):
      for epoch in range(0, self.epochs):
          for images, labels in self.dataset:
              ...

      return self.model

    def serialize_local_model(self, model):
        ...

    def deserialize_global_model(self, global_model):
        ...

    def get_model_shape(self):
        ...

    def get_model_dtype(self):
        ...

  # Construct the client layer.
  client = FLClient(...)

Aggregation Server
------------------

Modalic provides a lightweight server application which the Python SDK compliments. The server is modular,
which allows for integrating the server with your own custom API in any programming language.
The communication is handled via `gRPC <https://grpc.io/>`_ which is an open source high performance Remote Procedure Call (RPC)
framework that can run in any environment. It can efficiently connect services in and across data centers
with pluggable support for load balancing, tracing, health checking and authentication.

Even though the server is a separate software stack, it is possible to start the server via Python script from the SDK by running::

  modalic.run_server(conf_path="config.toml")

It is optional but recommended to add a `TOML <https://toml.io/en/>`_ configuration file which allows for
setting certain hyperparameters which control the Federated Learning process.
An `example config <https://github.com/modalic/python-sdk/blob/main/examples/pytorch_mnist/config.toml>`_ file can be found here.


Framework Support: Pytorch & TensorFlow
---------------------------------------

For now, the Modalic Python SDK offers full support for the two major Open-Source Machine Learning frameworks
`Pytorch <https://pytorch.org/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.

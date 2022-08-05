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

There are basically two main endpoints that enable the client's ability to participate in a Federated Learning
procedure. The two different possibilities of integrating FL into one's Machine Learning stack,
represent two different programming paradigms, one object-oriented and one functional.

If the object-oriented programming style is preferred, the :ref:`PytorchClient <modalic-pytorch-client>` and :ref:`TfClient <modalic-tf-client>` can be used.
If the functional paradigm is preferred, :ref:`@modalic.torch_train <modalic-torch-decor-apiref>` and :ref:`@modalic.tf_train <modalic-tf-decor-apiref>` can be applied.

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


Framework Support: Pytorch & Tensorflow
---------------------------------------

For now, the Modalic Python SDK offers full support for the two major Open-Source Machine Learning frameworks
`Pytorch <https://pytorch.org/>`_ and `Tensorflow <https://www.tensorflow.org/>`_.

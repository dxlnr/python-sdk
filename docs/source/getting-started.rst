.. _getting-started:

Getting Started
===============

Installation
------------

For installing the Python SDK and further information regarding the process,
please visit the `installation guide <getting-started>`_.

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


SDK Entrypoints
---------------

There are basically two main entrypoints that enable the client's ability to participate in a Federated Learning
procedure. The two different possibilities of integrating FL into one's Machine Learning stack,
represent two different programming paradigms, one object-oriented and one functional.



Frameworks Support: Pytorch
---------------------------

For now, the modalic Python SDK only offers full support for one of the major open source machine learning frameworks
`Pytorch <https://pytorch.org/>`_ but `Tensorflow <https://www.tensorflow.org/>`_ will come soon.

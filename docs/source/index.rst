=============
Documentation
=============

Modalic aims to provide an **Open-Source FLOps Platform** for performing Federated Learning reliably.
In order to foster adoption of Federated Learning, the creation of various kind of use cases and
ultimately bringing these into production, accessible technology needs to exist for fast
development, evaluation and testing.

Modalic introduces a technology core that includes two main components:

* A `Software Development Kit <python-sdk-ref>`_ written in Python that serves as an endpoint for the client (edge device) to participate in a Federated Learning setup.
  The Python SDK is compatible with the main Machine Learning frameworks like `PyTorch <https://pytorch.org/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.
  This allows for custom Machine Learning development without any compromise. The Modalic Python SDK serves then as an extra layer, abstracting functionality and prepares the client for collaboration.
  The collaboration is then coordinated by a central aggregation instance.
* An Aggregation Server application which can be directly called by `using the SDK <modalic-server-apiref>`_. The aggregation server is the backbone of Federated Learning
  and defines the algorithm of how the local models are compromised into a global model.

Get started using the :ref:`getting-started` or by reading about the :ref:`key concepts<concepts>`.

Content
-------

.. toctree::
    :maxdepth: 1

    installation
    getting-started
    tutorials-and-examples/index
    concepts
    sdk/index

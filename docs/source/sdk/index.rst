.. _python-sdk-ref:

Python SDK Reference
====================

The essential components provided by the Python SDK library are listed below. It supports currently the
two main Machine Learning Frameworks `Pytorch <https://github.com/pytorch/pytorch>`_ \& `Tensorflow <https://github.com/tensorflow/tensorflow>`_.
Each framework has their own API endpoint and can be used in object-oriented or procedural fashion.

.. _modalic-pytorch-client-apiref:

Pytorch Client SDK Modules
--------------------------

.. _modalic-pytorch-client:

modalic.PytorchClient
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: modalic.client.pytorch_client.PytorchClient
    :members:

.. _modalic-decor-apiref:

Functional Training Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: modalic.train


.. _modalic-tf-client-apiref:

Tensorflow Client SDK Modules
-----------------------------

.. _modalic-tf-client:

modalic.TfClient
~~~~~~~~~~~~~~~~
.. autoclass:: modalic.client.tf_client.TfClient
    :members:

.. _modalic-server-apiref:

The aggregation server can also be started from SDK.

Aggregation Server
------------------
.. autofunction:: modalic.run_server

.. _modalic-conf-apiref:

Configuration Object
--------------------
.. autofunction:: modalic.Conf

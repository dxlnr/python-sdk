.. _python-sdk-ref:

Python SDK Reference
====================

The essential components provided by the Python SDK library are listed below. It supports currently the
two main Machine Learning Frameworks `Pytorch <https://github.com/pytorch/pytorch>`_ \& `Tensorflow <https://github.com/tensorflow/tensorflow>`_.
Each framework has their own API endpoint and can be used in object-oriented or procedural fashion.

.. _modalic-pytorch-apiref:

Pytorch API
-----------

.. _modalic-pytorch-client:

modalic.PytorchClient
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: modalic.api.torch.pytorch_client.PytorchClient
    :members:

.. _modalic-torch-decor-apiref:

Functional Training Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: modalic.api.torch.torch_train


.. _modalic-tf-apiref:

Tensorflow API
--------------

.. _modalic-tf-client:

modalic.TfClient
~~~~~~~~~~~~~~~~
.. autoclass:: modalic.api.tf.tf_client.TfClient
    :members:

.. _modalic-tf-decor-apiref:

Functional Training Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: modalic.api.tf.tf_train


.. _modalic-server-apiref:

Aggregation Server
------------------
.. autofunction:: modalic.run_server

.. _modalic-conf-apiref:

Configuration Object
--------------------
.. autofunction:: modalic.Conf

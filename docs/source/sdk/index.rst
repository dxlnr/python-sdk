.. _python-sdk-ref:

Python SDK Reference
====================

The essential components provided by the Python SDK library are listed below. It supports currently the
two main Machine Learning Frameworks `Pytorch <https://github.com/pytorch/pytorch>`_ \& `TensorFlow <https://github.com/tensorflow/tensorflow>`_.
Each framework has their own API endpoint and can be used in object-oriented or procedural fashion.

.. _modalic-client-apiref:

Client API
-----------

.. _modalic-client:

modalic.Client
~~~~~~~~~~~~~~
.. autoclass:: modalic.client.client.Client
    :members:


Internal Client
---------------

.. _modalic-internal-client:

modalic.client.InternalClient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: modalic.client.internal_client.InternalClient
    :members:

Running the Client
------------------

.. _modalic-run-client:

modalic.run_client
~~~~~~~~~~~~~~~~~~
.. autofunction:: modalic.run.run_client


.. _modalic-server-apiref:

Aggregation Server
------------------
.. autofunction:: modalic.run_server

.. _modalic-conf-apiref:

Configuration Object
--------------------
.. autofunction:: modalic.Conf

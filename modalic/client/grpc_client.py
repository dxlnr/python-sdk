from abc import ABC, abstractmethod
from modalic.utils import protocol

class CommunicationLayer(ABC):
    r"""Abstract communicatio base layer for ensuring the grpc protocol.
    """

    @abstractmethod
    def update(self):
        r"""Sends an updated model version to the server."""
        raise NotImplementedError()

    @abstractmethod
    def get_global_model(self):
        r"""Client request to get the latest version of the global model
            from server.
        """
        raise NotImplementedError()


class Communicator(CommunicationLayer):
    r"""Communicator class object implements the grpc protocol functionality
    which will be inherited by some client class object.

    Args
    ----------
    """
    def __init__(self):
        pass

    @abstractmethod
    def set_weights(self, weights: protocol.Weights):
        r"""Set model weights from a list of NumPy ndarrays."""
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self) -> protocol.Weights:
        r"""Get model weights as a list of NumPy ndarrays."""
        raise NotImplementedError()

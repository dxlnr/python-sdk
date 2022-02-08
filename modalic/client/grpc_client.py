from abc import ABC, abstractmethod
from typing import Optional
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
    ------------------------------------------------------------
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

    def grpc_connection(self,
                        server_address: str,
                        max_message_length: int = 536870912,
                        root_certificates: Optional[bytes] = None):
        r"""Establishes a grpc connection to the server.
            Returns:
                (channel, stub): Tuple containing the thread-safe grpc channel
                to server & the grpc stub.
        """
        channel_options = [
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ]

        if root_certificates is not None:
            ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
            channel = grpc.secure_channel(
                server_address, ssl_channel_credentials, options=channel_options
            )
            # log(INFO, "Opened secure gRPC connection using certificates.")
        else:
            channel = grpc.insecure_channel(server_address, options=channel_options)
            # log(INFO, "Opened insecure gRPC connection.")
        return (channel, CommunicationStub(channel))

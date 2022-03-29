from abc import ABC, abstractmethod
from typing import Optional

import grpc

from modalic.utils import common
from modalic.client.proto.mosaic_pb2_grpc import CommunicationStub
from modalic.client.proto.mosaic_pb2 import ClientUpdate, ClientMessage


class CommunicationLayer(ABC):
    r"""Abstract communicatio base layer for ensuring the grpc protocol."""

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

        Args:
            server_address: static ip address of the aggregation server.
    """

    def __init__(self, server_address: str):
        self.server_address = server_address

    @abstractmethod
    def set_weights(self, weights: common.Weights):
        r"""Set model weights from a list of NumPy ndarrays."""
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self) -> common.Weights:
        r"""Get model weights as a list of NumPy ndarrays."""
        raise NotImplementedError()

    def grpc_connection(
        self,
        server_address: str,
        max_message_length: int = 536870912,
        root_certificates: Optional[bytes] = None,
    ):
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

    def update(self, dtype: str, round_id: int, stake: int, loss: float):
        r"""Sends an updated model version to the server.
        Args:
            dtype: Data Type of the trained model. Important as it determines the de-/serialization.
            round_id: Training round id.
            stake: Sets the number of samples the local model was trained on.
            loss: Loss of the local model during training.
        """
        weights = self.get_weights()
        parameters = parameters_to_proto(
            weights_to_parameters(weights, dtype=dtype, model_version=round_id)
        )
        process_meta = process_meta_to_proto(to_meta(round_id, loss))

        (channel, stub) = self.grpc_connection(self.server_address)
        send = stub.Update(
            ClientUpdate(
                id=self.cid,
                parameters=parameters,
                stake=stake,
                process_meta=process_meta,
            )
        )
        channel.close()

    def get_global_model(self):
        r"""Client request to get the latest version of the global model from server."""
        (channel, stub) = self.grpc_connection(self.server_address)
        response = stub.GetGlobalModel(ClientMessage(id=self.cid))

        params = parameters_from_proto(response)
        if not params.tensor:
            channel.close()
        else:
            weights = parameters_to_weights(params, self.model_shape)
            self.set_weights(weights)
            channel.close()

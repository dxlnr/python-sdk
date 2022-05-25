#  Copyright (c) modalic 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import INFO, WARNING
from typing import Any, List, Optional, Tuple

import grpc
import numpy as np

from modalic.client.proto.mosaic_pb2 import ClientMessage, ClientUpdate
from modalic.client.proto.mosaic_pb2_grpc import CommunicationStub
from modalic.client.utils.communication import _grpc_connection
from modalic.logging.logging import logger
from modalic.utils import shared
from modalic.utils.protocol import (
    parameters_from_proto,
    parameters_to_proto,
    process_meta_to_proto,
    to_meta,
)
from modalic.utils.serde import parameters_to_weights, weights_to_parameters


class CommunicationLayer(ABC):
    r"""Abstract communicatio base layer for ensuring the grpc protocol."""

    @abstractmethod
    def update(self, dtype: str, round_id: int, stake: int, loss: float) -> None:
        r"""Sends an updated model version to the server."""
        raise NotImplementedError()

    @abstractmethod
    def get_global_model(
        self, model_shape: List[np.ndarray[int, np.dtype[Any]]]
    ) -> None:
        r"""Client request to get the latest version of the global model
        from server.
        """
        raise NotImplementedError()


class Communicator(CommunicationLayer):
    r"""Communicator class object implements the grpc protocol functionality
    which will be inherited by some client class object.

    Args:
        server_address: static ip address of the aggregation server.
        cid: client identifier.
    """

    def __init__(self, server_address: str, cid: int):
        self.server_address = server_address
        self.cid = cid

    @abstractmethod
    def set_weights(self, weights: shared.Weights) -> None:
        r"""Set model weights from a list of NumPy ndarrays."""
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self) -> shared.Weights:
        r"""Returns the model weights as a list of NumPy ndarrays."""
        raise NotImplementedError()

    def grpc_connection(
        self,
        server_address: str,
        max_message_length: int = 536870912,
        root_certificates: Optional[bytes] = None,
        logging: Optional[bool] = False,
    ) -> Tuple[grpc.Channel, CommunicationStub]:
        r"""Establishes a grpc connection to the server.

        Args:
            server_address: Determines the IP address for connecting to the server.
            max_message_length: Maximum grpc message size.
            root_certificates: (optional) Can be set in order to establish a encrypted connection
                               between client & server.
            logging: (optional) bool for setting logging or not. Default: False

        Returns:
            (channel, stub): Tuple containing the thread-safe grpc channel
            to server & the grpc stub.
        """
        return _grpc_connection(
            server_address, max_message_length, root_certificates, logging
        )

    def update(self, dtype: str, round_id: int, stake: int, loss: float) -> None:
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
        _ = stub.Update(
            ClientUpdate(
                id=self.cid,
                parameters=parameters,
                stake=stake,
                process_meta=process_meta,
            )
        )
        channel.close()
        logger.log(INFO, f"Client {self.cid} sent update to aggregation server.")

    def get_global_model(self, model_shape: list[np.ndarray]) -> None:
        r"""Client request to get the latest version of the global model from server.

        Args:
            model_shape: Holds the shape of the model architecture for serialization & deserialization.
        """
        (channel, stub) = self.grpc_connection(self.server_address)
        response = stub.GetGlobalModel(ClientMessage(id=self.cid))

        params = parameters_from_proto(response)
        if not params.tensor:
            channel.close()
            logger.log(
                WARNING,
                f"Client {self.cid} did not receive global model from aggregation server.",
            )
        else:
            weights = parameters_to_weights(params, model_shape)
            self.set_weights(weights)
            channel.close()
            logger.log(
                INFO,
                f"Client {self.cid} received global model from aggregation server.",
            )

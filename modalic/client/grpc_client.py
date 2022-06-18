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
from logging import INFO
from typing import Any, List, Optional, Tuple

import grpc
import numpy as np

from modalic.client.proto.mosaic_pb2_grpc import CommunicationStub
from modalic.client.utils.communication import (
    _grpc_connection,
    _sync_model_version,
    _update,
)
from modalic.logging.logging import logger
from modalic.utils import shared
from modalic.utils.serde import parameters_to_weights


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
        cid: client identifier via unique integer.
    """

    def __init__(self, server_address: str, cid: int):
        self.server_address = server_address
        self.cid = cid
        self._round_id = 0

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
        logback: Optional[bool] = False,
    ) -> Tuple[grpc.Channel, CommunicationStub]:
        r"""Establishes a grpc connection to the server.

        Args:
            server_address: Determines the IP address for connecting to the server.
            max_message_length: Maximum grpc message size.
            root_certificates: (optional) Can be set in order to establish a encrypted connection
                               between client & server.
            logback: (optional) bool for setting logging or not. Default: False

        Returns:
            (channel, stub): Tuple containing the thread-safe grpc channel
            to server & the grpc stub.
        """
        return _grpc_connection(
            server_address, max_message_length, root_certificates, logback, self.cid
        )

    def update(self, dtype: str, round_id: int, stake: int, loss: float) -> None:
        r"""Sends an updated model version to the server.

        Args:
            dtype: Data Type of the trained model. Important as it determines the de-/serialization.
            round_id: Training round id.
            stake: Sets the number of samples the local model was trained on.
            loss: Loss of the local model during training.
        """
        _update(
            self.cid,
            self.server_address,
            self.get_weights(),
            dtype,
            round_id,
            stake,
            loss,
        )

    def get_global_model(
        self, model_shape: list[np.ndarray], retry: float = 5.0
    ) -> None:
        r"""Client request to get the latest version of the global model from server.

        Args:
            model_shape: Holds the shape of the model architecture for serialization & deserialization.
        """
        params = _sync_model_version(
            self.cid, self.server_address, self._round_id, retry_period=retry
        )
        if params is not None:
            weights = parameters_to_weights(params, model_shape)
            self.set_weights(weights)
            logger.log(
                INFO,
                f"Client {self.cid} received global model from aggregation server.",
            )

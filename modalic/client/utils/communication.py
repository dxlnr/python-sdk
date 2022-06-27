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

import time
from logging import ERROR, INFO, WARNING
from typing import Any, Optional, Tuple

import grpc

from modalic.client.proto.mosaic_pb2 import ClientMessage, ClientUpdate
from modalic.client.proto.mosaic_pb2_grpc import CommunicationStub
from modalic.logging.logging import logger
from modalic.utils import shared
from modalic.utils.protocol import (
    parameters_from_proto,
    parameters_to_proto,
    process_meta_to_proto,
    to_meta,
)
from modalic.utils.serde import weights_to_parameters


def _grpc_connection(
    server_address: str,
    max_message_length: int = 536870912,
    root_certificates: Optional[bytes] = None,
    logback: Optional[bool] = False,
    client_id: Optional[int] = 0,
) -> Tuple[grpc.Channel, CommunicationStub]:
    r"""Establishes a grpc connection to the server.

    Args:
        server_address: Determines the IP address for connecting to the server.
        max_message_length: Maximum grpc message size. Default: 536870912 which are 512MB : 512 * 1024 * 1024
        root_certificates: (optional) Can be set in order to establish a encrypted connection
                           between client & server. Default: None
        logback: (optional) bool for setting logging or not. Default: False
        client_id: (optional) Client ID used for logging purposes.

    Returns:
        (channel, stub): Tuple containing the thread-safe grpc channel
        to server & the grpc stub.
    """
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if root_certificates is not None:
        # with open('server.crt') as f:
        #     trusted_certs = f.read().encode()
        ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=channel_options
        )
        if logback:
            logger.log(
                INFO, "Client {} established secure gRPC connection.".format(client_id)
            )
    else:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        if logback:
            logger.log(
                INFO,
                f"Client {client_id} established insecure gRPC connection.",
            )
    return (channel, CommunicationStub(channel))


def _error_grpc(rpc_error: grpc.RpcError, **kwargs: dict[str, Any]) -> None:
    r"""Common grpc error message when exception is thrown.

    Args:
        rpc_error: grpc.RpcError which defines the kind of error message.
    """
    if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
        logger.log(
            ERROR,
            f"Aggregation server could not be reached. Please validate if server is up and running\
            and the IP {kwargs['server_address']} is correct.",
        )
        return
    else:
        logger.log(
            ERROR,
            f"Received RPC error: code={rpc_error.code()} message={rpc_error.details()}",
        )
        return


def _update(
    client_id: int,
    server_address: str,
    weights: shared.Weights,
    dtype: str,
    round_id: int,
    stake: int,
    loss: float,
) -> None:
    r"""Sends an updated model version to the server.

    Args:
        client_id: Client identifier
        server_address: Determines the IP address for connecting to the server.
        weights: The local model weights which are sent to the aggregation server.
        dtype: Data Type of the trained model. Important as it determines the de-/serialization.
        round_id: Training round id.
        stake: Sets the number of samples the local model was trained on.
        loss: Loss of the local model during training.
    """
    parameters = parameters_to_proto(
        weights_to_parameters(weights, dtype=dtype, model_version=round_id)
    )
    process_meta = process_meta_to_proto(to_meta(round_id, loss))

    (channel, stub) = _grpc_connection(server_address)
    try:
        _ = stub.Update(
            ClientUpdate(
                id=client_id,
                parameters=parameters,
                stake=stake,
                process_meta=process_meta,
            )
        )
    except grpc.RpcError as rpc_error:
        _error_grpc(rpc_error, server_address=server_address)
    else:
        channel.close()
        logger.log(INFO, f"Client {client_id} sent update to aggregation server.")


def _get_global_model(client_id: int, server_address: str) -> shared.Parameters:
    r"""Client request to get the latest version of the global model from server.

    Args:
        client_id: Client identifier
        server_address: Determines the IP address for connecting to the server.
    """
    (channel, stub) = _grpc_connection(server_address)
    try:
        response = stub.GetGlobalModel(ClientMessage(id=client_id))
    except grpc.RpcError as rpc_error:
        _error_grpc(rpc_error, server_address=server_address)
    else:
        channel.close()
        return parameters_from_proto(response)


def _sync_model_version(
    client_id: int,
    server_address: str,
    round_id: int,
    n_retry: int = 1,
    retry_period: float = 10.0,
    **kwargs: dict[str, Any],
) -> shared.Parameters:
    r"""Checks and syncs model version to current training round.

    Args:
        client_id: Client identifier
        server_address: Determines the IP address for connecting to the server.
        round_id: Curretn training round id for checking the global model version.
        n_retry: Number of retries that should be performed before ignoring
                 getting an updated global model.
        retry_period: Defines how long a logstep should take before retrying.
    """
    # TODO: This function need testing and rewriting.
    for n in range(n_retry):
        params = _get_global_model(client_id, server_address)
        if params is not None:
            if not params.tensor:
                logger.log(
                    WARNING,
                    f"Client {client_id} did not receive global model from aggregation server.",
                )
                return
            else:
                if params.model_version == round_id:
                    return params
                else:
                    logger.log(
                        INFO,
                        f"Client {client_id} in training round {round_id} received global model version {params.model_version}.\
                        Client goes in lockstep for {retry_period}s.",
                    )
                    time.sleep(retry_period)
        else:
            return

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

from logging import INFO
from typing import Optional, Tuple

import grpc

from modalic.client.proto.mosaic_pb2_grpc import CommunicationStub
from modalic.logging.logging import logger


def _grpc_connection(
    server_address: str,
    max_message_length: int = 536870912,
    root_certificates: Optional[bytes] = None,
    logback: Optional[bool] = False,
    cid: Optional[int] = 0,
) -> Tuple[grpc.Channel, CommunicationStub]:
    r"""Establishes a grpc connection to the server.

    Args:
        server_address: Determines the IP address for connecting to the server.
        max_message_length: Maximum grpc message size. Default: 536870912 which are 512MB : 512 * 1024 * 1024
        root_certificates: (optional) Can be set in order to establish a encrypted connection
                           between client & server. Default: None
        logback: (optional) bool for setting logging or not. Default: False
        cid: (optional) Client ID used for logging purposes.

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
                INFO, "Client {} established secure gRPC connection.".format(cid)
            )
    else:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        if logback:
            logger.log(
                INFO, "Client {} established insecure gRPC connection.".format(cid)
            )
    return (channel, CommunicationStub(channel))


# def _error_grpc(func(*args, **kwargs)):
#     r"""."""
#     try:
#         response = func(args)
#     except grpc.RpcError as rpc_error:
#         if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
#             logger.log(ERROR, f"Aggregation server could not be reached. Please validate IP {self.server_address}.")
#         else:
#             logger.log(ERROR, f"Received unknown RPC error: code={rpc_error.code()} message={rpc_error.details()}")
#
#     return response

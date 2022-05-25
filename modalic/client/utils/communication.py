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
    self,
    server_address: str,
    max_message_length: int = 536870912,
    root_certificates: Optional[bytes] = None,
    logging: Optional[bool] = False,
) -> Tuple[grpc.Channel, CommunicationStub]:
    r"""Establishes a grpc connection to the server.

    Args:
        server_address: Determines the IP address for connecting to the server.
        max_message_length: Maximum grpc message size. Default: 536870912 which are 512MB : 512 * 1024 * 1024
        root_certificates: (optional) Can be set in order to establish a encrypted connection
                           between client & server. Default: None
        logging: (optional) bool for setting logging or not. Default: False

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
        if logging:
            logger.log(
                INFO, "Client {} established secure gRPC connection.".format(self.cid)
            )
    else:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        if logging:
            logger.log(
                INFO, "Client {} established insecure gRPC connection.".format(self.cid)
            )
    return (channel, CommunicationStub(channel))

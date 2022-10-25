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
"""Spawn Client."""
from typing import Optional

from modalic.client import Client, InternalClient
from modalic.config import Conf


def spawn_client(client: Client, conf: Optional[Conf] = None):
    r"""
    :param client:
    """
    # Internal Client which implements the backend Federated protocol logic.
    #
    modalic_client = InternalClient(client, conf)
    # Spawns the InternalClient in a separate thread by using threading library.
    # `start` calls the `run` method of `InternalClient`.
    #
    # https://docs.python.org/3.8/library/threading.html#threading.Thread.start
    # https://docs.python.org/3.8/library/threading.html#threading.Thread.run
    modalic_client.start()

    # Join the main thread.
    #
    # This blocks the calling main thread until the thread whole `join` method
    # is called terminates.
    try:
        modalic_client.join()
    except KeyboardInterrupt:
        modalic_client.stop()

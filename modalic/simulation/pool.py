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

import concurrent.futures
import traceback
from typing import Any


class ClientPool:
    f"""Object holds and manages a bunch of individual simulated clients.

    Args:
        client: Modalic client object. Options are PytorchClient || TensorflowClient
        num_clients: Number of clients you want to run the federated learning with.
    """

    def __init__(self, client: Any, num_clients: int = 1):
        self.client = client
        self.num_clients = num_clients

    def run(self) -> None:
        r"""Endpoint to execute the whole client pool in parallel."""
        self.spawn_pool(self.num_clients)

    def spawn_pool(self, max_workers: int = 1) -> None:
        r"""Launching a pool of separated clients using concurrent.futures ThreadPoolExecutor."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.exec_single_thread, range(1, max_workers + 1))

    def exec_single_thread(self, name: str) -> None:
        r"""Executes the single thread object which holds the main functionality."""
        try:
            self.client.run()
        except Exception:
            traceback.print_exc()

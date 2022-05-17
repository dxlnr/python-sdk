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

import threading
import traceback

from sdk.client.client import Client


class ClientThread(threading.Thread):
    r"""Creates a thread that simulates a single client.

    Args:
        single_client: modalic client object.
    """

    def __init__(
        self, single_client: Client,
    ):
        self.single_client = single_client

    def run(self):
        r"""runs the simulation of a single client within separate thread."""
        try:
            self._run()
        except Exception as err:
            traceback.print_exc()

    def _run(self):
        counter = 0
        while not self._exit_event.is_set():
            counter += 1
            self.single_client.get_global_model(self.single_client.model_shape)
            self.single_client.train()
            self.single_client.update(self.single_client.dtype, counter, 0, 0)

            # time.sleep(60)
            if counter == 20:
                self.stop()

    def _stop(self):
        self._exit_event.set()

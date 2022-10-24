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

"""Base Client Object."""
import threading
from abc import abstractmethod
from typing import Any, List, Optional

from modalic.client.mosaic_python_sdk import mosaic_python_sdk


class Client(threading.Thread):
    def __init__(
        self,
        server_address: str,
        client: Any,
        state: Optional[List[int]] = None,
        scalar: float = 1.0,
    ):
        # Internal client
        #
        # Implements the internal logic that makes the client able to
        # process the Modalic Federated Learning protocol.
        self._mosaic_client = mosaic_python_sdk.Client(server_address, scalar, state)

        # Client API
        #
        # https://github.com/python/cpython/blob/3.9/Lib/multiprocessing/process.py#L80
        # stores the Client class with its args and kwargs.
        #
        # Based on composition over inheritance.
        self._client = client
        # Instantiate global model to None.
        self._global_model = None
        # State instigating that an error occured while fetching a global model
        # from the aggregation server.
        self._error_on_fetch_global_model = False
        # threading internals
        self._exit_event = threading.Event()
        # self._poll_period = Backoff(min_ms=100, max_ms=100000, factor=2, jitter=False)

        # Primitive lock objects. Once a thread has acquired a lock,
        # subsequent attempts to acquire it block, until it is released;
        # any thread may release it.
        self._step_lock = threading.Lock()
        super().__init__(daemon=True)

    @abstractmethod
    def serialize_local_model(self) -> list:
        r"""
        Serializes the local model into a `list` data type. The data type of the
        elements must match the data type attached as metadata.
        :returns: The local model (self.model) as a `list`.
        """
        raise NotImplementedError()

    @abstractmethod
    def deserialize_global_model(self, global_model: list) -> Any:
        r"""
        Deserializes the `global_model` from a `list` to a specific model type.
        The data type of the elements matches the data type defined.
        If no global model exists (usually in the first round), the method is
        not called.

        :param global_model: The global model.
        :returns:
        """
        raise NotImplementedError()

    @abstractmethod
    def on_new_global_model(self, model):
        r"""."""
        raise NotImplementedError()

    def _fetch_global_model(self):
        r"""
        :raises GlobalModelUnavailable:
        :raises GlobalModelDataTypeMisMatch:
        """
        try:
            global_model = self._mosaic_client.global_model()
        except (
            mosaic_python_sdk.GlobalModelUnavailable,
            mosaic_python_sdk.GlobalModelDataTypeError,
        ) as err:
            print(f"{err}")
            self._error_on_fetch_global_model = True
        else:
            if global_model is not None:
                self._global_model = self._client.deserialize_global_model(global_model)
            else:
                self._global_model = None
            self._error_on_fetch_global_model = False

    def _set_local_model(self, local_model: list):
        r"""Sets a local model. This method can be called at any time. Internally the
        participant first caches the local model.

        If a local model is already in the cache and `set_local_model` is called with a new local
        model, the current cached local model will be replaced by the new one.

        :param local_model: The local model in the form of a list. The data type of the
            elements must match the data type defined in the coordinator configuration.
        :raises LocalModelLengthMisMatch: If the length of the local model does
            not match the length defined in the coordinator configuration.
        :raises LocalModelDataTypeMisMatch: If the data type of the local model
            does not match the data type.
        """
        try:
            self._mosaic_client.set_model(local_model)
        except (mosaic_python_sdk.UninitializedClient) as err:
            print(f"{err}")
            self._exit_event.set()

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

from dataclasses import dataclass
from logging import WARNING
from typing import Any

import toml

from modalic.logging.logging import logger


@dataclass
class Conf(object):
    r"""Configuration object class that stores the parameters regarding the federated learning process.

    Args:
        server_address: GRPC endpoint for aggregation server.
        client_id: Client identifier which must be unique.
        timeout: Defines a timeout length in seconds which is mainly used for
                 simulating some waiting periode after each training round.
        training_rounds: Number of training rounds that should be performed.
        participants: Number of required clients (edge device) participating in a single training round.
        data_type: Models data type which defines the (de-)serialization of the model.

    Examples:
        >>> conf = Conf.create_conf({})
    """
    server_address: str = "[::]:8080"
    client_id: int = 0
    timeout: float = 0.0
    training_rounds: int = 0
    participants: int = 0
    data_type: str = "F32"

    def set_params(self, conf: dict[str, dict[str, Any]]) -> None:
        r"""Overwrites default parameters with external is stated.

        Args:
            conf: Produced by .toml config. Dict which contains dicts. The values
                  of conf will overwrite the default values.
        """
        if conf is not None:
            if value := self._find_keys(conf, "server_address"):
                self.server_address = value
            if value := self._find_keys(conf, "client_id"):
                self.client_id = value
            if value := self._find_keys(conf, "timeout"):
                self.timeout = value
            if value := self._find_keys(conf, "training_rounds"):
                self.training_rounds = value
            if value := self._find_keys(conf, "participants"):
                self.participants = value
            if value := self._find_keys(conf, "data_type"):
                self.data_type = value

    def _find_keys(self, blob: dict[str, dict[str, Any]], key_str: str = "") -> Any:
        r"""Finds the value for certain key in dictionary with arbitrary depth.

        Args:
            blob: Dictionary which is searched for the key value pair.
            key_str: Key that is searched for.

        Returns:
            Any value that belongs to the key.
        """
        value = None
        for (k, v) in blob.items():
            if k == key_str:
                return v
            if isinstance(v, dict):
                value = self._find_keys(v, key_str)
        return value

    @classmethod
    def create_conf(cls, conf: dict[str, dict[str, Any]] = None) -> Conf:
        r"""Constructs a (default) conig object with external conf if given.

        Args:
            conf: Produced by .toml config. Dict which contains dicts. The values
                  of conf will overwrite the default values.
        """
        instance = cls()
        instance.set_params(conf)
        return instance

    @classmethod
    def from_toml(cls, path: str) -> Conf:
        r"""Constructs a conig object from external .toml configuration file.

        Args:
            path: String path to .toml config file.
        """
        instance = cls()
        try:
            instance.set_params(toml.load(path))
        except FileNotFoundError:
            logger.log(
                WARNING,
                f"Config .toml via path '{path}' cannot be found. Default configuration parameters are used.",
            )
        return instance

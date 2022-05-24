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
from typing import Any


@dataclass
class Conf(object):
    r"""Configuration object class that stores the parameters regarding the federated learning process.

    Args:
        server_address: GRPC endpoint for aggregation server.
        timeout: Defines a timeout length in seconds which is mainly used for
                 simulating some waiting periode after each training round.
        training_rounds: Number of training rounds that should be performed.
    """
    server_address: str = "[::]:8080"
    timeout: float = 30.0
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
            for key, value in conf.items():
                if "address" in value.keys():
                    self.server_address = value["address"]
                if "timeout" in value.keys():
                    self.timeout = value["timeout"]
                if "rounds" in value.keys():
                    self.training_rounds = value["rounds"]
                if "participants" in value.keys():
                    self.participants = value["participants"]
                if "data_type" in value.keys():
                    self.data_type = value["data_type"]

    @classmethod
    def create_conf(cls, conf: dict[str, dict[str, Any]]) -> Conf:
        r"""Constructs a conig object with external conf.

        Args:
            conf: Produced by .toml config. Dict which contains dicts. The values
                  of conf will overwrite the default values.
        """
        instance = cls()
        instance.set_params(conf)
        return instance

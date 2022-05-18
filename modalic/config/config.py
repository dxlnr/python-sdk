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

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ConfigObject:
    r"""Configuration object class that stores the parameters regarding the federated learning process.

    Args:
        server_address:
        timeout:
    """
    server_address: str = "[::]:8000"
    timeout: float = 60.0
    training_rounds: int = 1

    def __init__(self, conf: dict[str, Any]):
        self.server_address = conf["server_address"]
        self.timeout = conf["timeout"]
        self.training_rounds = conf["rounds"]

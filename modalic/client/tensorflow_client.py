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

from collections import OrderedDict
from typing import Generic
import numpy as np
import tensorflow


class TensorflowClient(Communicator):
    r"""
    Tensorflow compatible client object which abstracts all the distributed communication.
    Serves as a simple layer and API that enables participating within a
    Federated Learning process.

    Parameters:
    ------------------------
        trainer: Tensorflow Trainer object.
        cid: Client id which uniquely identifies the client within the process.
        server_address: GRPC server address
    """

    def __init__(self,
                 trainer: Generic,
                 cid: int,
                 server_address: str,
    ):
        super().__init__(server_address)
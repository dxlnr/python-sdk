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
import sys

# Version bump
from ._version import __version__

# Client APIs for main frameworks
if "torch" in sys.modules:
    from modalic.api.torch import PytorchClient

if "tensorflow" in sys.modules:
    from modalic.api.tf import TfClient

# Decorators
if "torch" in sys.modules:
    from modalic.api.torch import torch_train

if "tensorflow" in sys.modules:
    from modalic.api.tf import tf_train

# Client Endpoint
from modalic.client import Client

# Configuration
from modalic.config import Conf

# Invoke Aggregation Server
from modalic.server.server import run_server

# Simulation API
from modalic.simulation import ClientPool

# Function Endpoint.
from .run import run_client

# module level doc-string
__doc__ = """
modalic - Federated Learning Operations Platform
================================================

"""


__all__ = [
    "Client",
    "PytorchClient",
    "TfClient",
    "Conf",
    "ClientPool",
    "torch_train",
    "tf_train",
    "run_server",
    "run_client",
]

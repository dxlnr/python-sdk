import sys

# Client API
if "torch" in sys.modules:
    from modalic.client.pytorch_client import PytorchClient

if "tensorflow" in sys.modules:
    from modalic.client.tf_client import TfClient

from modalic.client.trainer import Trainer

# Decorators
from modalic.client.utils.decor import train

# Configuration
from modalic.config import Conf

# Invoke Aggregation Server
from modalic.server.server import run_server

# Simulation API
from modalic.simulation import ClientPool

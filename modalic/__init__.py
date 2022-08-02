import sys

# Client APIs for main frameworks
if "torch" in sys.modules:
    from modalic.api.torch import PytorchClient

if "tensorflow" in sys.modules:
    from modalic.api.tf.tf_client import TfClient

# from modalic.client.trainer import Trainer

# Decorators
from modalic.api.torch.torch_func import train

# Configuration
from modalic.config import Conf

# Invoke Aggregation Server
from modalic.server.server import run_server

# Simulation API
from modalic.simulation import ClientPool

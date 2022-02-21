from modalic.client.grpc_client import Communicator
from modalic.utils import common

class Client(Communicator):
    r"""
    Client object which abstracts all the communication to Communicator.
    Serves as an simple layer and API that enables participating within
    Federated Learning process without designing the endpoints.

        Args:
            cfg: Configuration setup defined by .toml
    """
    def __init__(self,
                 cfg):
        super().__init__()

    def set_weights(self, weights: common.Weights)):
        r"""Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_weights(self):
        r"""Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def run(self):
        r"""Runs the whole process for a single modalic client."""
        pass

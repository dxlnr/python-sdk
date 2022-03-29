from modalic.client.grpc_client import Communicator
from modalic.utils import common


class Client(Communicator):
    r"""
    Client object which abstracts all the communication to Communicator.
    Serves as an simple layer and API that enables participating within
    Federated Learning process without designing the endpoints.

    Args:
        trainer: Pytorch Trainer object.
        cid: Client id which uniquely identifies the client within the process.
        server_address: GRPC server address
    """

    def __init__(self, trainer, cid: int, server_address: str):
        super().__init__(server_address)

        self.trainer = trainer
        self.cid = cid

    def set_weights(self, weights: common.Weights):
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

    def get_weights(self):
        r"""Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_model_shape(self):
        r"Extracts the shape of the pytorch model." ""
        shapes = list()
        for param_tensor in self.model.state_dict().keys():
            shapes.append(np.array(self.model.state_dict()[param_tensor].size()))
        return shapes

    def get_model_dtype(self):
        r"""Extracts the data type of the pytorch model."""
        torch_type = list(self.trainer.model.state_dict().items())[0][1].dtype
        if torch_type == "torch.float32" or "torch.float":
            dtype = "F32"
        elif torch_type == "torch.float64" or "torch.double":
            dtype = "F64"
        else:
            raise ValueError(
                "{} is not supported by aggregation server. Federation will fail. Please use 'torch.float' or 'torch.double'".format(
                    torch_type
                )
            )
        return dtype

    def train(self):
        self.model, self.loss = self.trainer.train(self.cid)

    def val_get_global_model(self, params: common.Parameters) -> bool:
        r"""Validates the response from server."""
        pass

    def run(self):
        r"""Runs the whole process for a single modalic client."""
        pass

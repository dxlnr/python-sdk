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

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from modalic.client.grpc_client import Communicator
from modalic.utils import common


class PytorchClient(Communicator):
    r"""
    Pytorch compatible client object which abstracts all the distributed communication.
    Serves as a simple layer and API that enables participating within a
    Federated Learning process.

    Args:
        trainer: Pytorch Trainer object.
        cid: Client id which uniquely identifies the client within the process.
        server_address: GRPC server address
    """

    def __init__(
        self, trainer: Any, cid: int, server_address: str,
    ):
        super().__init__(server_address, cid)
        self.cid = cid
        self.trainer = trainer
        self.model = self.trainer.model

        self.model_shape = self.get_model_shape()
        self.dtype = self.get_model_dtype()
        self.round_id = 0
        self.loss = 0.0

    def __repr__(self):
        return f"Modalic Pytorch Client Object {self.cid}"

    def set_weights(self, weights: common.Weights) -> None:
        r"""Set model weights from a list of NumPy ndarrays.

        Args:
            weights: Model weights as a list of NumPy ndarrays.
        """
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_weights(self) -> common.Weights:
        r"""Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_model_shape(self) -> list[np.ndarray]:
        r"Extracts the shape of the pytorch model." ""
        shapes: list[np.ndarray] = list()
        for param_tensor in self.model.state_dict().keys():
            shapes.append(np.array(self.model.state_dict()[param_tensor].size()))
        return shapes

    def get_model_dtype(self) -> str:
        r"""Extracts the data type of the pytorch model.

        Returns:
            dtype: Encodes the data type of the model as a String. Options are
                   "F32" and "F64".
        """
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

    def train(self) -> None:
        self.model, self.loss = self.trainer.train()

    # def eval(self) -> None:
    #     pass

    # def val_get_global_model(self, params: common.Parameters) -> bool:
    #     r"""Validates the response from server."""

    def _run(self) -> None:
        r"""Runs a single trainings round for a single modalic client."""
        self.round_id += 1
        self.get_global_model(self.model_shape)
        self.train()
        self.update(self.dtype, self.round_id, len(self.trainer.dataset), self.loss)

    # def run(self) -> None:
    #     r"""Looping the whole process for a single modalic client."""

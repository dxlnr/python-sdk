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

# import time
import traceback
from collections import OrderedDict
from logging import DEBUG, INFO
from typing import Any, Optional

import numpy as np
import torch

from modalic.client.grpc_client import Communicator
from modalic.config import Conf
from modalic.logging.logging import logger
from modalic.utils import shared


class PytorchClient(Communicator):
    r"""
    Pytorch compatible client object which abstracts all the distributed communication.
    Serves as a simple layer and API that enables participating within a
    Federated Learning process.

    Args:
        trainer: Pytorch Trainer object.
        data: Dataset object that can be set for the Pytorch Trainer object.
        conf: Configuration object that stores all the parameters concerning the process.
        cid: Client id which uniquely identifies the client within the process.

    Examples:
        >>> client = modalic.PytorchClient(Trainer(), 1)
        >>> client.run()
    Raises:
        AttributeError: Input object trainer has to contain a model & train() function.
    """

    def __init__(
        self,
        trainer: Any,
        cid: Optional[int] = 0,
        conf: Optional[dict] = None,
        # data: Optional[Any] = None,
    ):
        self.trainer = trainer
        self.conf = Conf.create_conf(conf)
        self.cid = cid

        super().__init__(self.conf.server_address, self.cid)

        try:
            self.model = self.trainer.model
        except AttributeError:
            traceback.print_exc()

        # Setting all the internal necessary attributes.
        self._training_rounds = self.conf.training_rounds
        self._model_shape = self._get_model_shape()
        self._get_model_dtype()
        if hasattr(self.trainer, "dataset"):
            if isinstance(self.trainer.dataset, list):
                self._data_size = len(self.trainer.dataset)
            else:
                self._data_size = 1
        else:
            logger.log(
                DEBUG,
                f"Object {self.trainer} has no attribute dataset.\
                Federation will proceed with default value 1 as the size of the dataset.",
            )
            self._data_size = 1

        self._loss = 0.0

    def __repr__(self) -> str:
        return f"Modalic Pytorch Client Object {self.cid}"

    @property
    def dtype(self):
        r"""."""
        return self._dtype

    @property
    def round_id(self):
        r"""."""
        return self._round_id

    @property
    def loss(self):
        r"""."""
        return self._loss

    @property
    def model_shape(self):
        r"""."""
        return self._model_shape

    def set_weights(self, weights: shared.Weights) -> None:
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

    def get_weights(self) -> shared.Weights:
        r"""Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _get_model_shape(self) -> list[np.ndarray]:
        r"""Extracts the shape of the pytorch model."""
        shapes: list[np.ndarray] = list()
        for param_tensor in self.model.state_dict().keys():
            shapes.append(np.array(self.model.state_dict()[param_tensor].size()))
        return shapes

    def _get_model_dtype(self) -> None:
        r"""Extracts the data type of the pytorch model.

        Returns:
            dtype: Encodes the data type of the model as a String. Options are
                   "F32" and "F64".
        """
        torch_type = list(self.trainer.model.state_dict().items())[0][1].dtype
        if torch_type == "torch.float32" or "torch.float":
            self._dtype = "F32"
        elif torch_type == "torch.float64" or "torch.double":
            self._dtype = "F64"
        else:
            raise ValueError(
                f"{torch_type} is not supported by aggregation server. \
                Federation will fail. Please use 'torch.float' or 'torch.double'."
            )

    def _train(self) -> None:
        r"""Runs the train method of custom trainer object for single model."""
        try:
            self.model, self._loss = self.trainer.train()
        except AttributeError:
            raise AttributeError(f"{self.trainer} has no train() functionality.")

    # def eval(self) -> None:
    #     pass

    # def val_get_global_model(self, params: shared.Parameters) -> bool:
    #     r"""Validates the response from server."""

    def _run(self) -> None:
        r"""Runs a single trainings round for a single modalic client."""
        self.get_global_model(self._model_shape)
        self._round_id += 1
        self._train()
        logger.log(
            INFO,
            f"Client {self.cid} | training round: {self._round_id} | loss: {self._loss}",
        )
        self.update(self._dtype, self._round_id, self._data_size, self._loss)
        # time.sleep(self.conf.timeout)

    def run(self) -> None:
        r"""Looping the whole process for a single modalic client."""
        while self._round_id < self._training_rounds:
            self._run()

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

import time
import traceback
from logging import DEBUG, INFO
from typing import Any, List, Optional

import numpy as np

from modalic.client.grpc_client import Communicator
from modalic.client.utils.torch_utils import (
    _get_model_dtype,
    _get_model_shape,
    _get_torch_weights,
    _set_torch_weights,
)
from modalic.config import Conf
from modalic.data.misc import get_dataset_length
from modalic.logging.logging import logger
from modalic.utils import shared


class PytorchClient(Communicator):
    r"""
    Pytorch compatible client object which abstracts all the distributed communication.
    Serves as a simple layer and API that enables participating within a
    Federated Learning process.

    :param trainer: Pytorch Trainer object.
    :param conf: Configuration object that stores all the parameters concerning the process.
    :param data: Dataset object that can be set for the Pytorch Trainer object.
    :param client_id: Client id which uniquely identifies the client within the process.

    Examples:
        >>> client = modalic.PytorchClient(Trainer(), conf, 1)
        >>> client.run()

    :raises AttributeError: Input object trainer has to contain a model & train() function.
    """

    def __init__(
        self,
        trainer: Any,
        conf: Optional[dict] = None,
        client_id: Optional[int] = 0,
        # data: Optional[Any] = None,
    ):
        self.trainer = trainer
        self.conf = Conf.create_conf(conf)

        if client_id != 0:
            self.client_id = client_id
            self.conf.client_id = client_id
        else:
            self.client_id = self.conf.client_id

        super().__init__(self.conf.server_address, self.client_id)

        try:
            self.model = self.trainer.model
        except AttributeError:
            traceback.print_exc()

        # Setting all the internal necessary attributes.
        self._training_rounds = self.conf.training_rounds
        self._model_shape = self._get_model_shape()
        self._get_model_dtype()
        if hasattr(self.trainer, "dataset"):
            self._data_size = get_dataset_length(self.trainer.dataset)
        else:
            logger.log(
                DEBUG,
                f"Object {self.trainer} has no attribute dataset.\
                Federation will proceed with default value 1 as the size of the dataset.",
            )
            self._data_size = 1

        self._loss = 0.0

    def __repr__(self) -> str:
        r"""Returns string representative of object."""
        return f"Modalic Pytorch Client Object {self.client_id}"

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

    # def _validate_trainer(self):
    #     r"""Raises exception if trainer object does not contain certain attributes
    #     and functionalities.
    #     """
    #     pass

    def _set_weights(self, weights: shared.Weights) -> None:
        r"""Sets model weights from a list of NumPy ndarrays.

        :param weights: Model weights as a list of NumPy ndarrays.
        """
        self.model = _set_torch_weights(self.model, weights)

    def _get_weights(self) -> shared.Weights:
        r"""Returns model weights as a list of NumPy ndarrays."""
        return _get_torch_weights(self.model)

    def _get_model_shape(self) -> List[np.ndarray]:
        r"""Extracts the shape of the pytorch model.

        :returns: List of np.array representing the model shape.
                  Example: [np.array([1, 4]), np.array([1])]
        """
        return _get_model_shape(self.model)

    def _get_model_dtype(self) -> None:
        r"""Extracts the data type of the pytorch model."""
        self._dtype = _get_model_dtype(self.model)

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

    def _run_single_round(self) -> None:
        r"""Runs a single trainings round for a single modalic client."""
        self.get_global_model(self._model_shape)
        self._round_id += 1
        self._train()
        logger.log(
            INFO,
            f"Client {self.client_id} | training round: {self._round_id} | loss: {self._loss}",
        )
        self.update(self._dtype, self._round_id, self._data_size, self._loss)
        time.sleep(self.conf.timeout)

    def train(self) -> None:
        r"""Looping the whole training process for a single modalic client."""
        while self._round_id < self._training_rounds:
            self._run_single_round()

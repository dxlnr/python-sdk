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

from typing import Any, List, Optional

import numpy as np

from modalic.client.client import Client
from modalic.client.utils.torch_utils import (
    _get_tf_model_dtype,
    _get_tf_model_shape,
    _get_tf_weights,
    _set_tf_weights,
)
from modalic.config import Conf
from modalic.utils import shared


class TfClient(Client):
    r"""."""

    def __init__(
        self,
        trainer: Any,
        conf: Optional[dict] = None,
        # data: Optional[Any] = None,
    ):
        self.trainer = trainer
        if conf is None:
            self.conf = Conf.create_conf(conf)
        else:
            self.conf = conf

        try:
            self.model = self.trainer.model
        except AttributeError:
            raise AttributeError(
                f"Custom {trainer} object has no model. Please define the model architecture that should be trained."
            )

        super().__init__(
            self.trainer, self.conf, self._get_model_shape(), self._get_model_dtype()
        )

    def _set_weights(self, weights: shared.Weights) -> None:
        r"""Sets the model weights from a list of NumPy ndarrays.

        :param weights: Model weights as a list of NumPy ndarrays.
        """
        self.model = _set_tf_weights(self.model, weights)

    def _get_weights(self) -> shared.Weights:
        r"""Returns model weights as a list of NumPy ndarrays."""
        return _get_tf_weights(self.model)

    def _get_model_shape(self) -> List[np.ndarray]:
        r"""Extracts the shape of the tensorflow model.

        :returns: List of np.array representing the model shape.
            (Example: [np.array([1, 4]), np.array([1])])
        """
        return _get_tf_model_shape(self.model)

    def _get_model_dtype(self) -> None:
        r"""Extracts the data type of the tensorflow model."""
        self._dtype = _get_tf_model_dtype(self.model)

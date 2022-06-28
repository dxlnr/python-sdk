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
from typing import List

import numpy as np
import torch

from modalic.utils import shared


def _set_torch_weights(
    model: torch.nn.Module, weights: shared.Weights
) -> torch.nn.Module:
    r"""Set model weights from a list of NumPy ndarrays.

    Args:
        model: Pytorch model object.
        weights: Model weights as a list of NumPy ndarrays.

    Returns:
        model: Pytorch model object that is updated with input weights,
    """
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def _get_torch_weights(model: torch.nn.Module) -> shared.Weights:
    r"""Get model weights as a list of NumPy ndarrays.

    Args:
        model: Pytorch model object.

    Returns:
        model: Weights as a list of NumPy ndarrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _get_model_dtype(model: torch.nn.Module) -> str:
    r"""Extracts the data type of the pytorch model.

    Args:
        model: Pytorch model object.

    Returns:
        dtype: Encodes the data type of the model as a String. Options are
               "F32" and "F64".

    Raises:
        ValueError: If torch type is not either 'torch.float' or 'torch.double'.
    """
    torch_type = list(model.state_dict().items())[0][1].dtype
    if torch_type == "torch.float32" or "torch.float":
        _dtype = "F32"
    elif torch_type == "torch.float64" or "torch.double":
        _dtype = "F64"
    else:
        raise ValueError(
            f"{torch_type} is not supported by aggregation server. \
            Federation will fail. Please use 'torch.float' or 'torch.double'."
        )
    return _dtype


def _get_model_shape(model: torch.nn.Module) -> List[np.ndarray]:
    r"""Extracts the shape of the pytorch model.

    Args:
        model: Pytorch model object.

    Returns:
        shape: List of np.ndarray which contains the shape (size) of each individual layer of the model.
    """
    return [
        np.array(model.state_dict()[param_tensor].size())
        for param_tensor in model.state_dict().keys()
    ]

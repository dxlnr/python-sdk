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
"""Torch specific Helper Functions"""
import itertools

import numpy as np
import torch.nn as nn


def _np_ndarray_to_float(ndarray: np.ndarray) -> list:
    r"""Serialize NumPy ndarray to list of u8 bytes."""
    return [float(single) for single in np.nditer(ndarray)]


def _ser_np_weights(weights: list) -> list:
    r"""Serialize NumPy ndarray to bytes."""
    layers = [_np_ndarray_to_float(ndarray) for ndarray in weights]
    return list(itertools.chain(*layers))


def serialize_torch_model(model: nn.Module) -> list:
    """Serializes torch model as `py list`.

    :param model: Torch model.
    :returns: One level list of py floats.
    """
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return _ser_np_weights(weights)

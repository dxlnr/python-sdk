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

import numpy as np
import tensorflow as tf
import torch

from modalic.client.utils.tf_utils import _get_tf_model_dtype, _get_tf_weights
from modalic.client.utils.torch_utils import _get_torch_model_dtype, _get_torch_weights
from modalic.utils.serde import (
    _bytes_to_ndarray,
    _dtype_to_struct,
    _weights_to_bytes,
    get_shape,
)


def test_serialisation_deserialisation() -> None:
    r"""Test if after serialization/deserialisation the np.ndarray is identical."""
    arg = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

    serialized = _weights_to_bytes(arg, "!f")
    deserialized = _bytes_to_ndarray(serialized, get_shape(arg), "!f")

    # Assert deserialized array is equal to original
    np.testing.assert_equal(deserialized, arg)


def test_serialisation_deserialisation_w_torch(torch_model: torch.nn.Module) -> None:
    r"""Testing the serialization/deserialisation process of a pytorch model.

    :param torch_model: Arbitray simple torch model which the test is examined on.
    """
    weights = _get_torch_weights(torch_model)

    # Hyperparameters
    dtype = _get_torch_model_dtype(torch_model)
    assert dtype == "F32"

    dstruct = _dtype_to_struct(dtype)
    assert dstruct == "!f"

    model_shape = get_shape(weights)

    serialized = _weights_to_bytes(weights, dstruct)
    deserialized = _bytes_to_ndarray(serialized, model_shape, dstruct)

    np.testing.assert_equal(deserialized, weights)


def test_serialisation_deserialisation_w_keras(keras_model: tf.keras.Model) -> None:
    r"""Testing the serialization/deserialisation process of a keras model.

    :param keras_model: Arbitray simple keras model which is used for testing.
    """
    weights = _get_tf_weights(keras_model)

    print("type of keras: ", type(keras_model))
    print(isinstance(keras_model, tf.keras.Model))

    # Hyperparameters
    dtype = _get_tf_model_dtype(keras_model)
    print("in serde: ", dtype)
    assert dtype == "F32"

    dstruct = _dtype_to_struct(dtype)
    assert dstruct == "!f"

    model_shape = get_shape(weights)

    serialized = _weights_to_bytes(weights, dstruct)
    deserialized = _bytes_to_ndarray(serialized, model_shape, dstruct)

    np.testing.assert_equal(deserialized, weights)

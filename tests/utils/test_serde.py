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

from modalic.utils.serde import _bytes_to_ndarray, get_shape, weights_to_bytes


def test_serialisation_deserialisation() -> None:
    r"""Test if after serialization/deserialisation the np.ndarray is
    identical."""
    arg = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

    serialized = weights_to_bytes(arg, "!f")
    deserialized = _bytes_to_ndarray(serialized, get_shape(arg), "!f")

    # Assert deserialized array is equal to original
    np.testing.assert_equal(deserialized, arg)


# def test_serialisation_deserialisation_w_arg(
#     arg: np.ndarray, shape: List, dtype: str
# ) -> None:
#     r"""Testing the serialization/deserialisation process of the models.
#         Args:
#             input (np.ndarray): Tested array.
#     """
#     serialized = weights_to_bytes(arg, dtype)
#     deserialized = bytes_to_ndarray(serialized, shape, dtype)
#
#     # Assert deserialized array is equal to original
#     np.testing.assert_equal(deserialized, arg)

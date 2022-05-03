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

"""ProtoBuf serialization and deserialization."""
import itertools
import struct
import typing
from typing import Any, Iterable, List

import numpy as np

from modalic.utils import common, protocol


@typing.no_type_check
def weights_to_parameters(
    weights: common.Weights, dtype: str, model_version: int
) -> protocol.Parameters:
    r"""Convert NumPy weights to parameters object."""
    tensor = weights_to_bytes(weights, dtype_to_struct(dtype))
    return protocol.Parameters(
        tensor=tensor, data_type=dtype, model_version=model_version
    )


@typing.no_type_check
def parameters_to_weights(
    parameters: protocol.Parameters, shapes: List[np.ndarray[int, np.dtype[Any]]]
) -> common.Weights:
    r"""Convert parameters object to NumPy weights."""
    return bytes_to_ndarray(
        parameters.tensor, shapes, dtype_to_struct(parameters.data_type)
    )


def ndarray_to_bytes(ndarray: np.ndarray[Any, np.dtype[Any]], dtype: str) -> List[int]:
    r"""Serialize NumPy ndarray to list of u8 bytes."""
    res: List[int] = list()
    for single in np.nditer(ndarray):
        res.extend(struct.pack(dtype, single))
    return res


def weights_to_bytes(weights: common.Weights, dtype: str) -> bytes:
    r"""Serialize NumPy ndarray to bytes."""
    layers = [ndarray_to_bytes(ndarray, dtype) for ndarray in weights]
    return bytes(list(itertools.chain(*layers)))


def bytes_to_ndarray(
    tensor: bytes, layer_shape: List[np.ndarray[int, np.dtype[Any]]], dtype: str
) -> common.Weights:
    r"""Deserialize NumPy ndarray from u8 bytes."""
    layer: List[Any] = list()
    if dtype == "!f":
        for content in chunk(tensor, 4):
            layer.append(struct.unpack(">f", bytes(content)))
    elif dtype == "!d":
        for content in chunk(tensor, 8):
            layer.append(struct.unpack(">d", bytes(content)))
    else:
        raise TypeError("data type {} is not known.".format(dtype))

    layers = np.split(np.array(layer), indexing([np.prod(s) for s in layer_shape]))

    return [np.reshape(layer, shapes) for layer, shapes in zip(layers, layer_shape)]


def get_shape(weights: common.Weights) -> List[Any]:
    r"""Returns the shape of weights."""
    return [np.array(layer.size) for layer in weights]


def chunk(iterable: Iterable[Any], chunksize: int) -> zip[Any]:
    r"""helper chunking an iterable."""
    return zip(*[iter(iterable)] * chunksize)


def indexing(length: List[int]) -> List[int]:
    r"""helper for preparing the indices at which array is splitted."""
    for idx, _ in enumerate(length):
        if idx == 0:
            continue
        if (idx - 1) == len(length):
            break
        length[idx] += length[idx - 1]
    return length


def dtype_to_struct(dtype: str) -> str:
    r"""Prepare dtype for conversion with struct."""
    if dtype == "F32":
        return "!f"
    elif dtype == "F64":
        return "!d"
    else:
        raise TypeError("data type {} is not known.".format(dtype))

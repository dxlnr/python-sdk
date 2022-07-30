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

# from typing import List
#
# import numpy as np
# import tensorflow as tf
#
# from modalic.utils import shared
#
#
# def _set_tf_weights(
#     model: tf.keras.Model, weights: shared.Weights
# ) -> tf.keras.Model:
#     r"""Set model weights from a list of NumPy ndarrays.
#
#     :param model: Tensorflow model object.
#     :param weights: Model weights as a list of NumPy ndarrays.
#     :returns: Tensorflow model object that is updated with input weights.
#     """
#     pass
#
#
# def _get_tf_weights(model: tf.keras.Model) -> shared.Weights:
#     r"""Get model weights as a list of NumPy ndarrays.
#
#     :param model: Tensorflow model object.
#     :returns: Weights as a list of NumPy ndarrays.
#     """
#     pass
#
#
# def _get_tf_model_dtype(model: tf.keras.Model) -> str:
#     r"""Extracts the data type of the Tensorflow model.
#
#     :param model: Tensorflow model object.
#     :returns: dtype: Encodes the data type of the model as a String. Options are
#         "F32" and "F64".
#     :raises ValueError:
#     """
#     pass
#
#
# def _get_tf_model_shape(model: tf.keras.Model) -> List[np.ndarray]:
#     r"""Extracts the shape of the Tensorflow model.
#
#     :param model: Tensorflow model object.
#     :returns: List of np.ndarray which contains the shape (size) of each individual layer of the model.
#     """
#     pass

# def _det_torch(model: any) -> bool:
#     r"""Determines wether the model is a Tensorflow model or not."""
#     return isinstance(model, tf.keras.Model)

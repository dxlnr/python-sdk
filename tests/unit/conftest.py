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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
import tensorflow as tf
import torch


def get_torch_model_definition():
    r"""
    Defines a PyTorch model class that inherits from ``torch.nn.Module``.
    This method can be invoked within a pytest fixture to define the model class in the ``__main__`` scope.
    Alternatively, it can be invoked within a module to define the class in the module's scope.
    """

    # pylint: disable=W0223
    class SubclassedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x):
            # pylint: disable=arguments-differ
            y_pred = self.linear(x)
            return y_pred

    return SubclassedModel


@pytest.fixture(scope="module")
def torch_model():
    r"""
    A custom PyTorch model inheriting from ``torch.nn.Module`` whose class is defined in the
    "__main__" scope.
    """
    model_class = get_torch_model_definition()
    model = model_class()
    # train_model(model=model, data=data)
    yield model


def get_keras_model_definition():
    r"""
    Defines a PyTorch model class that inherits from ``tf.keras.Model``.
    This method can be invoked within a pytest fixture to define the model class in the ``__main__`` scope.
    Alternatively, it can be invoked within a module to define the class in the module's scope.
    """

    class SubclassedModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
            self.max1 = tf.keras.layers.MaxPooling2D(3)
            self.bn1 = tf.keras.layers.BatchNormalization()

            self.dense = tf.keras.layers.Dense(5)

        def call(self, input_tensor, training=False):
            x = self.conv1(input_tensor)
            x = self.max1(x)
            x = self.bn1(x)

            return self.dense(x)

    return SubclassedModel


@pytest.fixture(scope="module")
def keras_model():
    r"""
    A custom tf keras model inheriting from ``tf.keras.Model`` whose class is defined in the
    "__main__" scope.
    """
    model_class = get_keras_model_definition()
    model = model_class()
    # train_model(model=model, data=data)
    yield model

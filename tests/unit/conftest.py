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

import pytest
import torch


def get_torch_model_definition():
    """
    Defines a PyTorch model class that inherits from ``torch.nn.Module``. This method can be invoked
    within a pytest fixture to define the model class in the ``__main__`` scope. Alternatively, it
    can be invoked within a module to define the class in the module's scope.
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
    """
    A custom PyTorch model inheriting from ``torch.nn.Module`` whose class is defined in the
    "__main__" scope.
    """
    model_class = get_torch_model_definition()
    model = model_class()
    # train_model(model=model, data=data)
    yield model

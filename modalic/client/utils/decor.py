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

from logging import INFO

from modalic.client.utils.communication import _get_global_model, _update
from modalic.client.utils.torch_utils import (
    _get_model_shape,
    _get_torch_weights,
    _set_torch_weights,
)
from modalic.logging.logging import logger
from modalic.utils.serde import parameters_to_weights


def train(func):
    r"""Training function decorator. Performs the underlying train() function
    multiply times while participating in a federated learning procedure.

    Examples:
        >>> @modalic.train
    """

    def wrapper(model, *args, **kwargs):
        wrapper.model_shape = _get_model_shape(model)

        while wrapper.round_id < 10:
            params = _get_global_model(wrapper.client_id, wrapper.server_address)
            print("params: ", params)

            if params is not None and len(params.tensor) != 0:
                weights = parameters_to_weights(params, wrapper.model_shape)
                model = _set_torch_weights(model, weights)
                logger.log(
                    INFO,
                    f"Client {wrapper.client_id} received global model from aggregation server.",
                )
            wrapper.round_id += 1

            print(f"calling train in round {wrapper.round_id}")
            # print(f"model: {model[-1]}")

            model = func(model, *args, **kwargs)

            logger.log(
                INFO,
                f"Client {wrapper.client_id} | training round: {wrapper.round_id} | loss: {wrapper.loss}",
            )
            _update(
                wrapper.client_id,
                wrapper.server_address,
                _get_torch_weights(model),
                "F32",
                wrapper.round_id,
                1,
                wrapper.loss,
            )

    wrapper.round_id = 0
    wrapper.loss = 0.0
    wrapper.client_id = 1
    wrapper.server_address = "[::]:8080"

    return wrapper

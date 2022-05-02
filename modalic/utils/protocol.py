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

"""gRPC action script."""

from modalic.client.proto.mosaic_pb2 import (
    ClientMessage,
    Parameters,
    ProcessMeta,
)

from modalic.utils import common


def parameters_to_proto(parameters: common.Parameters) -> Parameters:
    r"""."""
    return Parameters(
        tensor=parameters.tensor,
        data_type=parameters.data_type,
        model_version=parameters.model_version,
    )


def parameters_from_proto(msg: Parameters) -> common.Parameters:
    r"""."""
    tensor: List[bytes] = list(msg.parameters.tensor)
    return common.Parameters(
        tensor=tensor,
        data_type=msg.parameters.data_type,
        model_version=msg.parameters.model_version,
    )


def process_meta_to_proto(meta: common.ProcessMeta) -> ProcessMeta:
    r"""."""
    return ProcessMeta(round_id=meta.round_id, loss=meta.loss)


def to_meta(round_id: int, loss: float) -> common.ProcessMeta:
    r"""."""
    return common.ProcessMeta(round_id=round_id, loss=loss)

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

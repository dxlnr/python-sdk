from typing import List
from dataclasses import dataclass
import numpy as np

Weights = List[np.ndarray]


@dataclass
class Parameters:
    r"""Model parameters."""

    tensor: bytes
    data_type: str
    model_version: int


@dataclass
class ProcessMeta:
    r"""Meta data about the process."""

    round_id: int
    loss: float

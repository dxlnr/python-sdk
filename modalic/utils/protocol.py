# from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np

Weights = List[np.ndarray]

@dataclass
class Parameters:
    """Model parameters."""
    tensor: bytes
    data_type: str
    model_version: int

import numpy as np
# from typing import List

from modalic.utils.serde import weights_to_bytes, bytes_to_ndarray, get_shape


def test_serialisation_deserialisation() -> None:
    """Test if after serialization/deserialisation the np.ndarray is
    identical."""
    arg = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

    serialized = weights_to_bytes(arg, "!f")
    deserialized = bytes_to_ndarray(serialized, get_shape(arg), "!f")

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

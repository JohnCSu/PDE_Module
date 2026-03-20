from .dummy_types import *
from warp.types import vector, matrix
from collections.abc import Iterable


def get_unique_key(dictionary: dict, base_name: str) -> str:
    """Generate a unique key for a dictionary by appending a counter if needed."""
    candidate = base_name
    counter = 1
    while candidate in dictionary:
        candidate = f"{base_name}_{counter}"
        counter += 1
    return candidate


class SignatureMismatchError(ValueError):
    """Raised when a provided function does not meet the required input structure."""

    pass


def tuplify(x: Any) -> tuple:
    """Convert a single item into a tuple.

    Strings and bytes are treated as a single object.
    All other iterables are converted to tuple.
    """
    if isinstance(x, (str, bytearray, bytes)):
        return (x,)
    elif isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def dtype_from_shape(shape: int | Iterable[int], float_dtype) -> wp_Vector | wp_Matrix:
    """Return a warp vector or matrix dtype based on the shape provided.

    If shape is an int or tuple of length 1, returns a vector.
    If shape is a tuple of length 2, returns a matrix.
    """
    if isinstance(shape, Iterable):
        assert all(isinstance(x, int) for x in shape), (
            "contents in input/output must be int only"
        )
    else:
        assert isinstance(shape, int)
        shape = tuplify(shape)

    if len(shape) == 1:
        return vector(length=shape[0], dtype=float_dtype)
    else:
        return matrix(shape=shape, dtype=float_dtype)

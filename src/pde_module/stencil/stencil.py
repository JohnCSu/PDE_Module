import warp as wp
from warp import types
from ..utils.dummy_types import *

from collections.abc import Iterable
from typing import Any, Optional


def tuplify(x: Any) -> tuple:
    """Convert single item into tuple.

    String and bytes are treated as a single object.
    All other iterables are converted to tuple.
    """
    if isinstance(x, (str, bytearray, bytes)):
        return (x,)
    elif isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class Stencil:
    """Array of Structure Stencil Base Module.

    Similar to nn.Module from PyTorch, this class:
    - Identifies the appropriate dtype for input and output
    - Initializes and stores the output array

    To use, subclass and implement the `forward` method.
    """

    initial: bool = True

    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def create_output_array(
        input_array: wp.array, output_dtype: Optional[types] = None
    ) -> wp_Array:
        """Create an output array based on the input array and target dtype.

        If output_dtype is None, the input_array dtype is used.
        Note: the returned array is not zeroed.

        Args:
            input_array: The input wp.array to determine shape from.
            output_dtype: Optional target dtype for the output array.

        Returns:
            A new wp.array with the same shape and specified dtype.
        """
        shape = input_array.shape
        output_dtype = output_dtype if output_dtype is not None else input_array.dtype
        return wp.empty(shape=shape, dtype=output_dtype)

    def forward(self, *args, **kwargs) -> Any:
        """Compute the stencil output.

        This method must be implemented by subclasses. It should contain
        all the code necessary to calculate the stencil output.
        """
        raise NotImplementedError(
            "forward method in Stencil Object must be implemented by the user"
        )

    def _set_registry(self, call: str) -> str:
        """Set up and return the name of the registry list for a given call type."""
        call_list_name = f"_{call}_list"
        if hasattr(self, call_list_name) is False:
            call_list = []

            for attr in dir(self):
                method = getattr(self, attr)
                if hasattr(method, f"_{call}_order") and callable(method):
                    call_list.append(method)

            setattr(
                self,
                call_list_name,
                sorted(call_list, key=lambda x: getattr(x, f"_{call}_order")),
            )

        return call_list_name

    def setup(self, *args, **kwargs) -> None:
        """Initialize the stencil safely outside of __call__ to ensure initial flag is set."""
        _ = self._set_registry("before_forward")
        _ = self._set_registry("after_forward")

        call_list_name = self._set_registry("setup")

        for method in getattr(self, call_list_name):
            method(*args, **kwargs)

        self.initial = False

    def before_forward(self, *args, **kwargs) -> None:
        """Hook to perform operations before forward pass.

        By default, this method is empty. Override to add pre-processing.
        """
        call_list_name = self._set_registry("before_forward")
        for method in getattr(self, call_list_name):
            method(*args, **kwargs)

    def after_forward(self, *args, **kwargs) -> None:
        """Hook to perform operations after forward pass.

        By default, this method does nothing.
        """
        call_list_name = self._set_registry("after_forward")
        for method in getattr(self, call_list_name):
            method(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Call the stencil, executing the full pipeline.

        Steps:
            1. Setup (if initial == True, otherwise skip)
            2. Before Forward Call
            3. Forward Call - compute results (must be implemented)
            4. After Forward Call
            5. Return contents from forward call
        """
        if self.initial:
            self.setup(*args, **kwargs)
        self.before_forward(*args, **kwargs)
        output = self.forward(*args, **kwargs)
        self.after_forward(*args, **kwargs)
        return output


if __name__ == "__main__":
    x = wp.array()
    wp.types.vector()

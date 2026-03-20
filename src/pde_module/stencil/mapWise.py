from .stencil import Stencil
import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from .hooks import *
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils import *

class MapWise(Stencil):
    """Base class for map-wise operations on arrays.

    The first input must be an array and the output array is assumed to have
    the same dtype and shape as the input.
    """

    def __init__(self, element_op) -> None:
        self.element_op = element_op
        super().__init__()

    @setup(order=1)
    def initialize_kernel(self, array_A: wp.array, *args, **kwargs) -> None:
        self.input_dtype = array_A.dtype
        self.output_dtype = array_A.dtype
        self.output_array = self.create_output_array(array_A)
        self.kernel = wp.map(
            self.element_op, array_A, *args, out=self.output_array, return_kernel=True
        )

    def forward(self, array_A: wp.array, *args, **kwargs) -> wp_Array:
        """Apply map-wise operation to array A.

        Args:
            array_A: Input array.

        Returns:
            Output array with same shape and dtype as array_A.
        """
        dim = array_A.shape
        wp.launch(
            self.kernel,
            dim=dim,
            inputs=[array_A, *args],
            outputs=[self.output_array],
        )
        return self.output_array

from .stencil import Stencil
import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from .hooks import *
from ..utils import *

class ElementWise(Stencil):
    """Base class for element-wise operations between two arrays of the same size.

    Element-wise operations apply a function to each pair of corresponding
    elements in two arrays.
    """

    def __init__(self, element_op, output_dtype=None) -> None:
        self.element_op = element_op
        self.output_dtype = output_dtype
        super().__init__()

    @setup(order=1)
    def initialize_kernel(
        self, array_A: wp.array, array_B: wp.array, *args, **kwargs
    ) -> None:
        assert array_A.shape == array_B.shape, "input arrays must be the same Shape!"
        self.array_B_dtype = array_B.dtype
        self.array_A_dtype = array_A.dtype

        if self.output_dtype is None:
            self.output_dtype = self.array_A_dtype

        self.kernel = create_ElementOp_kernel(
            self.element_op, array_A.dtype, array_B.dtype, self.output_dtype
        )
        self.output_array = self.create_output_array(array_A, self.output_dtype)

    def forward(
        self, array_A: wp.array, array_B: wp.array, *args, **kwargs
    ) -> wp_Array:
        """Apply element-wise operation to arrays A and B.

        Args:
            array_A: First input array.
            array_B: Second input array with same shape as array_A.

        Returns:
            Output array containing the element-wise result.
        """
        dim = array_A.size
        wp.launch(
            self.kernel,
            dim=dim,
            inputs=[array_A.flatten(), array_B.flatten()],
            outputs=[self.output_array.flatten()],
        )
        return self.output_array


def create_ElementOp_kernel(
    element_op, array_A_dtype, array_B_dtype, output_array_dtype
):
    """Create a kernel for element-wise operations.

    Args:
        element_op: The element-wise operation function.
        array_A_dtype: Dtype of the first input array.
        array_B_dtype: Dtype of the second input array.
        output_array_dtype: Dtype of the output array.

    Returns:
        A wp.kernel that performs the element-wise operation.
    """

    @wp.kernel
    def elementWise_kernel(
        A: wp.array(dtype=array_A_dtype),
        B: wp.array(dtype=array_B_dtype),
        output_array: wp.array(dtype=output_array_dtype),
    ):
        i = wp.tid()
        output_array[i] = element_op(A[i], B[i])

    return elementWise_kernel

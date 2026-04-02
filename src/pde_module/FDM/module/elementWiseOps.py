from ...stencil.elementWise import ElementWise
import warp as wp
from warp.types import vector, matrix, types_equal
from ...stencil.hooks import *
from ...utils import dtype_from_shape, wp_Function, wp_Vector, wp_Matrix, wp_Array
from pde_module.FDM.kernel import scalarVectorMultiply
from pde_module.FDM.functional import elementWiseOp


class OuterProduct(ElementWise):
    """Compute the outer product between two vectors and output as a matrix.

    The outer product of vectors A and B produces a matrix where each element
    M[i,j] = A[i] * B[j].
    """

    def __init__(self, vec_A: int, vec_B: int, float_dtype=wp.float32) -> None:
        element_op = wp.outer
        output_dtype = matrix((vec_A, vec_B), dtype=float_dtype)
        super().__init__(element_op, output_dtype=output_dtype)
        self.vec_A = vector(vec_A, float_dtype)
        self.vec_B = vector(vec_B, float_dtype)

    @setup(order=2)
    def check_same(self, array_A: wp.array, array_B: wp.array, *args, **kwargs) -> None:
        """Verify that input arrays have the expected dtypes."""
        assert types_equal(array_A.dtype, self.vec_A) and types_equal(
            array_B.dtype, self.vec_A
        )

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
        return elementWiseOp(
            self.kernel,
            array_A,
            array_B,
            self.output_array,
            device=self.device,
        )


class scalarVectorMult(ElementWise):
    """Multiply a scalar field with a vector/matrix field.

    The scalar field should be a vector of length 1, and it multiplies
    element-wise with the second field which can be an arbitrary vector or matrix.
    """

    def __init__(self, outputs: int | tuple[int, ...], float_dtype=wp.float32) -> None:
        output_dtype = dtype_from_shape(outputs, float_dtype)
        element_op = scalarVectorMultiply(output_dtype)
        super().__init__(element_op, output_dtype)

    @setup(order=2)
    def check_same(self, array_A: wp.array, array_B: wp.array, *args, **kwargs) -> None:
        """Verify that input arrays have the expected dtypes."""
        assert array_A.dtype._length_ == 1 and types_equal(
            array_B.dtype, self.output_dtype
        )

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
        return elementWiseOp(
            self.kernel,
            array_A,
            array_B,
            self.output_array,
            device=self.device,
        )

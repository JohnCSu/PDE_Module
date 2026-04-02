import warp as wp
from warp.types import vector
from pde_module.utils import dtype_from_shape, wp_Function, wp_Vector, wp_Matrix


def scalarVectorMultiply(output_dtype: wp_Vector | wp_Matrix) -> wp_Function:
    """Create a function for scalar-vector multiplication.

    Args:
        output_dtype: The dtype of the non-scalar input.

    Returns:
        A wp.func that multiplies a scalar (length-1 vector) with the output_dtype field.
    """
    float_type = output_dtype._wp_scalar_type_

    @wp.func
    def _scalarVectorMultiply(
        a: vector(1, float_type), b: output_dtype
    ) -> output_dtype:
        return a[0] * b

    return _scalarVectorMultiply

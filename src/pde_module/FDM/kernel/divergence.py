import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from pde_module.stencil.utils import (
    create_stencil_op,
    eligible_dims_and_shift,
    create_tensor_divergence_op,
)
from pde_module.utils.types import *

def create_Divergence_kernel(
    input_dtype: wp_Vector | wp_Matrix,
    output_vector: vector,
    grid_shape: tuple[int, ...],
    stencil: vector,
    ghost_cells: int,
):
    """Create a kernel for computing the divergence.

    Args:
        input_dtype: Warp vector or matrix dtype for the input field.
        output_vector: Warp vector dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the divergence.
    """
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    if type_is_vector(input_dtype):
        div_op = create_stencil_op(input_dtype, stencil, ghost_cells)
    else:
        div_op = create_tensor_divergence_op(
            input_dtype, stencil, grid_shape, ghost_cells
        )

    @wp.kernel
    def Divergence_kernel(
        input_values: wp.array3d(dtype=input_dtype),
        alpha: input_dtype._wp_scalar_type_,
        output_values: wp.array3d(dtype=output_vector),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        div = output_vector()
        if wp.static(type_is_matrix(input_dtype)):
            div = div_op(input_values, index, stencil)
        else:
            for i in range(wp.static(len(dims))):
                div[0] += div_op(input_values, index, stencil, dims[i])[i]

        div *= alpha

        output_values[index[0], index[1], index[2]] = div

    return Divergence_kernel

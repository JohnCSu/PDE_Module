import warp as wp
from warp.types import vector, type_is_vector, type_is_matrix
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Kernel


def create_Grad_kernel(
    input_vector: wp_Vector,
    output_dtype: wp_Vector | wp_Matrix,
    grid_shape: tuple[int, ...],
    stencil: wp_Vector,
    ghost_cells: int,
) -> wp_Kernel:
    """Create a kernel for computing the gradient.

    Args:
        input_vector: Warp vector dtype for the input field.
        output_dtype: Warp vector or matrix dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the gradient.
    """
    assert type_is_vector(input_vector)

    output_dtype_is_vector = type_is_vector(output_dtype)
    assert type_is_vector(output_dtype) or type_is_matrix(output_dtype)

    stencil_op = create_stencil_op(input_vector, stencil, ghost_cells)
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    @wp.kernel
    def grad_kernel(
        scalar_array: wp.array3d(dtype=input_vector),
        alpha: input_vector._wp_scalar_type_,
        grad_array: wp.array3d(dtype=output_dtype),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        grad = output_dtype()

        for i in range(wp.static(len(dims))):
            if wp.static(output_dtype_is_vector):
                grad[i] = stencil_op(scalar_array, index, stencil, dims[i])[0]
            else:
                grad[:, i] = stencil_op(scalar_array, index, stencil, dims[i])

        grad *= alpha

        grad_array[index[0], index[1], index[2]] = grad

    return grad_kernel

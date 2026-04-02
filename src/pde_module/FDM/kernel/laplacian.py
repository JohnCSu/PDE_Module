import warp as wp
from warp.types import vector
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils.types import wp_Vector, wp_Kernel


def create_Laplacian_kernel(
    input_vector: vector, grid_shape: tuple[int, ...], stencil: vector, ghost_cells: int
) -> wp_Kernel:
    """Create a kernel for computing the Laplacian.

    Args:
        input_vector: Warp vector dtype for the input field.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the Laplacian.
    """
    assert wp.types.type_is_vector(input_vector), "Input type must be of vector"

    stencil_op = create_stencil_op(input_vector, stencil, ghost_cells)
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    @wp.kernel
    def laplacian_kernel(
        input_values: wp.array3d(dtype=input_vector),
        alpha: input_vector._wp_scalar_type_,
        output_values: wp.array3d(dtype=input_vector),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        laplace = input_vector()
        for i in range(wp.static(len(dims))):
            laplace += stencil_op(input_values, index, stencil, dims[i])

        laplace *= alpha

        output_values[index[0], index[1], index[2]] = laplace

    return laplacian_kernel

import warp as wp
from warp.types import vector, matrix
from ..utils.types import wp_Vector, wp_Vec3i


def create_stencil_op(input_vector: vector, stencil: vector, ghost_cells: int):
    """Create a stencil operation for finite difference calculations.

    Gathers neighbors of the current index along a given axis and applies
    the stencil weights. Outputs a vector of the same type as input_vector.

    Args:
        input_vector: Warp vector dtype for the input field.
        stencil: Warp vector dtype containing stencil weights.
        ghost_cells: Number of ghost cells in the grid.

    Returns:
        A wp.func that computes the stencil operation.
    """
    assert wp.types.type_is_vector(input_vector)
    length = stencil._length_
    assert (length % 2) == 1, "stencil must be odd sized"
    max_shift = (length - 1) // 2

    assert max_shift <= ghost_cells, (
        "Max shift must be <= ghost cell to avoid out of bounds array access!"
    )

    @wp.func
    def stencil_op(
        input_values: wp.array3d(dtype=input_vector),
        index: wp.vec3i,
        stencil: type(stencil),
        axis: int,
    ) -> input_vector:
        value = input_vector()
        for i in range(wp.static(length)):
            shift = i - max_shift
            stencil_index = index
            stencil_index[axis] = index[axis] + shift
            value += (
                input_values[stencil_index[0], stencil_index[1], stencil_index[2]]
                * stencil[i]
            )

        return value

    return stencil_op


def create_tensor_divergence_op(
    input_matrix: matrix, stencil: vector, grid_shape: tuple[int, ...], ghost_cells: int
):
    """Create a tensor divergence operation for matrix fields.

    Args:
        input_matrix: Warp matrix dtype for the input field.
        stencil: Warp vector dtype containing stencil weights.
        grid_shape: Shape of the grid (3-tuple).
        ghost_cells: Number of ghost cells in the grid.

    Returns:
        A wp.func that computes the tensor divergence.
    """
    N, D = input_matrix._shape_
    length = stencil._length_
    assert (length % 2) == 1, "stencil must be odd sized"
    max_shift = (length - 1) // 2
    assert max_shift <= ghost_cells, (
        "Max shift must be <= ghost cell to avoid out of bounds array access!"
    )

    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)
    assert D == len(eligible_dims), (
        "Dimensions of field and num col in matrix must match"
    )

    output_vec = vector(N, input_matrix._wp_scalar_type_)

    @wp.func
    def tensor_divergence(
        input_values: wp.array3d(dtype=input_matrix),
        index: wp.vec3i,
        stencil: type(stencil),
    ) -> output_vec:
        out = output_vec()
        for d in range(wp.static(len(eligible_dims))):
            axis = eligible_dims[d]
            for i in range(wp.static(length)):
                shift = i - max_shift
                stencil_index = index
                stencil_index[axis] = index[axis] + shift
                current_val_mat = input_values[
                    stencil_index[0], stencil_index[1], stencil_index[2]
                ]
                out += current_val_mat[:, axis] * stencil[i]

        return out

    return tensor_divergence


def eligible_dims_and_shift(
    grid_shape: tuple[int, ...], ghost_cells: int
) -> tuple[wp_Vector, wp_Vec3i]:
    """Return eligible dimensions and their corresponding ghost cell shifts.

    Returns dimensions that have more than 1 point as a vector, and the
    corresponding shift for each eligible dim due to ghost_cells.

    Args:
        grid_shape: Shape of the grid (3-tuple).
        ghost_cells: Number of ghost cells.

    Returns:
        Tuple of (eligible_dims vector, shift vector).

    Example:
        (3, 5, 1) with ghost_cells > 0 returns ((0, 1), (gc, gc, 0))
    """
    d = tuple(i for i, x in enumerate(grid_shape) if x > 1)
    return wp.types.vector(length=len(d), dtype=int)(d), wp.vec3i(
        [ghost_cells if x > 1 else 0 for x in grid_shape]
    )

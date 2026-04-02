import warp as wp
from warp.types import matrix
from pde_module.stencil.utils import eligible_dims_and_shift
from pde_module.utils.constants import INT32_MAX
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Kernel


def create_dampingLayer_kernel(
    input_dtype: wp_Vector | wp_Matrix,
    num_layers: int,
    p: float,
    grid_shape: tuple[int, ...],
    ghost_cells: int,
) -> wp_Kernel:
    """Create a kernel for the viscous damping layer.

    Args:
        input_dtype: Warp vector or matrix dtype.
        num_layers: Thickness of damping layer.
        p: Polynomial order for damping profile.
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel implementing the damping layer.
    """
    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)
    dimension = len(eligible_dims)
    limits = matrix(shape=(3, 2), dtype=int)()

    for i, s in enumerate(grid_shape):
        if s > 1:
            limits[i] = wp.vec2i([num_layers, (s - 1) - num_layers])

    float_type = input_dtype._wp_scalar_type_
    grid_shape_vec = wp.vec3i(grid_shape)

    @wp.func
    def get_argmin_and_min(grid_point: wp.vec3i) -> wp.int32:
        out = wp.vec3i()

        for i in range(3):
            boundary = grid_shape_vec[i]
            point = grid_point[i]
            out[i] = wp.where(
                boundary > 1, wp.min(point, boundary - point - 1), INT32_MAX
            )
        return wp.int32(wp.argmin(out))

    @wp.kernel
    def dampingLayer(
        du_dt: wp.array3d(dtype=input_dtype),
        grid_points: wp.array(dtype=wp.vec3i),
        C_max: float_type,
        output_array: wp.array3d(dtype=input_dtype),
    ):
        tid = wp.tid()

        grid_point = grid_points[tid]

        axis = get_argmin_and_min(grid_point)

        assert 0 <= axis < 2

        x = grid_point[axis]
        if (grid_shape_vec[axis] - 1 - x) < x:
            dx = x - limits[axis, 1]
        else:
            dx = limits[axis, 0] - x

        damp_factor = C_max * (float_type(dx) / float_type(num_layers)) ** p
        output_array[grid_point[0], grid_point[1], grid_point[2]] = (
            -damp_factor * du_dt[grid_point[0], grid_point[1], grid_point[2]]
        )

    return dampingLayer

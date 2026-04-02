import warp as wp
from pde_module.utils.types import wp_Array, wp_Matrix
from pde_module.LBM.utils import get_adjacent_ijk
from pde_module.utils import ijk_to_global_c


def create_streaming_kernel(
    f_arr: wp_Array,
    int_directions: wp_Matrix,
    num_distributions: int,
    grid_shape: tuple[int],
    dimension: int,
):
    grid_shape = wp.vec3i(grid_shape)

    adjacent_ijk = get_adjacent_ijk(dimension, grid_shape)

    @wp.kernel
    def streaming_kernel(
        f_in: wp.array2d(dtype=f_arr.dtype),
        f_out: wp.array2d(dtype=f_arr.dtype),
    ):
        i, j, k = wp.tid()

        global_id = ijk_to_global_c(
            i, j, k, grid_shape[0], grid_shape[1], grid_shape[2]
        )
        for f in range(num_distributions):
            vel_dir = -int_directions[f]
            ni, nj, nk = adjacent_ijk(i, j, k, vel_dir)
            adj_id = ijk_to_global_c(
                ni, nj, nk, grid_shape[0], grid_shape[1], grid_shape[2]
            )
            f_out[f, global_id] = f_in[f, adj_id]

    return streaming_kernel

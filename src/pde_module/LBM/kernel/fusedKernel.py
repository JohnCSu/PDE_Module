import warp as wp
from warp.types import vector, matrix
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Dtype
from pde_module.LBM.utils import get_adjacent_ijk, create_rho_and_u_func
from pde_module.utils import ijk_to_global_c, global_to_ijk_c
from pde_module.LBM.kernel.BGK import create_BGK_feq

FLUID = 0
SOLID_WALL = 1
MOVING_WALL = 2
EQUILIBRIUM = 3


def create_fusedLBMKernel(
    weights: wp_Vector,
    opposite_index: wp_Vector,
    int_directions: wp_Matrix,
    float_directions: wp_Matrix,
    num_distributions: int,
    grid_shape: tuple[int],
    dimension: int,
    float_dtype: wp_Dtype,
):
    grid_shape = wp.vec3i(grid_shape)
    adjacent_ijk = get_adjacent_ijk(dimension, grid_shape)
    calc_rho_and_u = create_rho_and_u_func(
        True, num_distributions, float_directions, dimension, float_dtype
    )
    BGK_feq = create_BGK_feq(dimension, float_dtype)

    @wp.kernel
    def fused_kernel(
        f_in: wp.array2d(dtype=float_dtype),
        inv_tau: float_dtype,
        flags: wp.array3d(dtype=wp.uint8),
        u_BC: wp.array3d(dtype=vector(dimension, float_dtype)),
        rho_BC: wp.array3d(dtype=float_dtype),
        f_out: wp.array2d(dtype=float_dtype),
    ):
        tid = wp.tid()
        global_id = tid
        i, j, k = global_to_ijk_c(
            global_id, grid_shape[0], grid_shape[1], grid_shape[2]
        )
        f_values = vector(float_dtype(0.0), length=num_distributions)

        for f in range(num_distributions):
            vel_dir = -int_directions[f]
            ni, nj, nk = adjacent_ijk(i, j, k, vel_dir)
            adj_id = ijk_to_global_c(
                ni, nj, nk, grid_shape[0], grid_shape[1], grid_shape[2]
            )
            opp_f = opposite_index[f]

            rho_wall = rho_BC[ni, nj, nk]
            u_wall = u_BC[ni, nj, nk]
            adj_flag = flags[ni, nj, nk]
            adj_f_in = f_in[f, adj_id]
            opp_f_in = f_in[opp_f, global_id]

            f_values[f] = wp.where(adj_flag == FLUID, adj_f_in, f_values[f])
            f_values[f] = wp.where(adj_flag == SOLID_WALL, opp_f_in, f_values[f])
            f_values[f] = wp.where(
                adj_flag == MOVING_WALL,
                opp_f_in
                + 2.0
                * 3.0
                * weights[f]
                * rho_wall
                * wp.dot(float_directions[f], u_wall),
                f_values[f],
            )

        rho, u = calc_rho_and_u(f_values, global_id)

        for f in range(num_distributions):
            feq = BGK_feq(weights[f], rho, u, float_directions[f])
            f_out[f, global_id] = f_values[f] - inv_tau * (f_values[f] - feq)

    return fused_kernel

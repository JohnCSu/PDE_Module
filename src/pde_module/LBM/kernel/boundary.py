import warp as wp
from warp.types import vector
from pde_module.utils.types import wp_Vector, wp_Matrix
from pde_module.LBM.utils import get_adjacent_ijk
from pde_module.utils import ijk_to_global_c, global_to_ijk_c
from pde_module.LBM.kernel.BGK import create_BGK_feq

FLUID = 0
SOLID_WALL = 1
MOVING_WALL = 2
EQUILIBRIUM = 3


def create_boundary_kernel(
    num_distributions,
    opposite_index,
    int_directions,
    float_directions,
    weights,
    dimension,
    grid_shape,
    sigma,
    float_dtype,
):
    grid_shape = wp.vec3i(grid_shape)
    adjacent_ijk = get_adjacent_ijk(dimension, grid_shape)
    BGK_feq = create_BGK_feq(dimension, float_dtype)

    @wp.func
    def get_adjacent_fluid(
        i: int,
        j: int,
        k: int,
        flags: wp.array3d(dtype=wp.uint8),
    ):
        idx_vec = vector(0, length=dimension)
        shift = wp.vec2i(-1, 1)
        for ii in range(dimension):
            for jj in range(2):
                idx_vec[ii] = shift[jj]
                ni, nj, nk = adjacent_ijk(i, j, k, idx_vec)
                if flags[ni, nj, nk] == FLUID:
                    return ni, nj, nk
                idx_vec[ii] = 0

        return i, j, k

    @wp.kernel
    def apply_BC(
        f_in: wp.array2d(dtype=float_dtype),
        flags: wp.array3d(dtype=wp.uint8),
        global_ids: wp.array1d(dtype=int),
        u_BCs: wp.array1d(dtype=vector(dimension, float_dtype)),
        rho_BCs: wp.array1d(dtype=float_dtype),
        f_out: wp.array2d(dtype=float_dtype),
    ):
        tid = wp.tid()
        global_id = global_ids[tid]
        u_wall = u_BCs[tid]
        rho_wall = rho_BCs[tid]
        i, j, k = global_to_ijk_c(
            global_id, grid_shape[0], grid_shape[1], grid_shape[2]
        )

        fluid_i, fluid_j, fluid_k = get_adjacent_fluid(i, j, k, flags)
        fluid_adj = ijk_to_global_c(
            fluid_i, fluid_j, fluid_k, grid_shape[0], grid_shape[1], grid_shape[2]
        )
        u = vector(float_dtype(0.0), length=dimension)
        rho = float_dtype(0.0)

        for f in range(num_distributions):
            rho += f_in[f, fluid_adj]
            u += f_in[f, fluid_adj] * float_directions[f]
        u /= rho

        u_wall = wp.where(wp.isnan(u_wall[0]), u, u_wall)
        rho_wall = wp.where(wp.isnan(rho_wall), rho, rho_wall)
        rho_wall = wp.abs(rho_wall)

        bc_is_moving = flags[i, j, k] == MOVING_WALL
        bc_is_equil = flags[i, j, k] == EQUILIBRIUM
        is_relaxed_equil = rho_wall < float_dtype(0.0)

        for f in range(num_distributions):
            opp_f = opposite_index[f]
            opp_vel = int_directions[opp_f]
            ni, nj, nk = adjacent_ijk(i, j, k, opp_vel)
            adj_is_fluid = flags[ni, nj, nk] == FLUID

            f_opp = f_in[opp_f, global_id]

            new_opp_f = wp.where(flags[ni, nj, nk] == FLUID, f_in[f, global_id], f_opp)
            new_opp_f -= wp.where(
                adj_is_fluid and bc_is_moving,
                2.0 * 3.0 * weights[f] * rho_wall * wp.dot(float_directions[f], u_wall),
                float_dtype(0.0),
            )
            new_opp_f = wp.where(
                bc_is_equil,
                BGK_feq(weights[opp_f], rho_wall, u_wall, float_directions[opp_f]),
                new_opp_f,
            )
            new_opp_f = wp.where(
                bc_is_equil and is_relaxed_equil,
                new_opp_f * sigma + (1.0 - sigma) * f_opp,
                new_opp_f,
            )
            f_out[opp_f, global_id] = new_opp_f

    return apply_BC

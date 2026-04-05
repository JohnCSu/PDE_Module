import warp as wp
from warp.types import vector, matrix
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Dtype
from pde_module.LBM.utils import get_adjacent_ijk, create_rho_and_u_func
from pde_module.utils import ijk_to_global_c, global_to_ijk_c
from pde_module.LBM.kernel.BGK import create_BGK_feq
from pde_module.LBM.flags import *


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
    calc_rho_and_u_vec = create_rho_and_u_func(
        True, num_distributions, float_directions, dimension, float_dtype
    )
    
    calc_rho_and_u_array = create_rho_and_u_func(
        False, num_distributions, float_directions, dimension, float_dtype
    )
    
    BGK_feq = create_BGK_feq(dimension, float_dtype)

    @wp.kernel
    def fused_kernel(
        f_in: wp.array2d(dtype=float_dtype),
        inv_tau: float_dtype,
        flags: wp.array3d(dtype=wp.uint8),
        u_BC: wp.array3d(dtype=vector(dimension, float_dtype)),
        rho_BC: wp.array3d(dtype=float_dtype),
        sigma:float_dtype,
        ramp:float_dtype,
        f_out: wp.array2d(dtype=float_dtype),
    ):
        tid = wp.tid()
        global_id = tid
        i, j, k = global_to_ijk_c(
            global_id, grid_shape[0], grid_shape[1], grid_shape[2]
        )
        f_values = vector(float_dtype(0.0), length=num_distributions)

        
        rho, u = calc_rho_and_u_array(f_in, global_id)
        
        
        
        for f in range(num_distributions):
            current_f = f_in[f,global_id]
            
            #Indices
            vel_dir = -int_directions[f]
            ni, nj, nk = adjacent_ijk(i, j, k, vel_dir)
            adj_id = ijk_to_global_c(
                ni, nj, nk, grid_shape[0], grid_shape[1], grid_shape[2]
            )
            opp_f = opposite_index[f]

            rho_wall = rho_BC[ni, nj, nk] # Equilibrium can be unknown
            u_wall = u_BC[ni, nj, nk]
            
            u_wall = wp.where(wp.isnan(u_wall[0]), u, ramp*u_wall)
            rho_wall = wp.where(wp.isnan(rho_wall), rho, rho_wall)
            
            adj_flag = flags[ni, nj, nk]
            adj_f_in = f_in[f, adj_id]
            opp_f_in = f_in[opp_f, global_id]
            
            is_relax = rho_wall < float_dtype(0.)
            rho_wall = wp.abs(rho_wall)
            
            new_f = float_dtype(0.)
            new_f = wp.where(adj_flag == FLUID, adj_f_in, new_f)
            new_f = wp.where(adj_flag == SOLID_WALL, opp_f_in, new_f)
            new_f = wp.where(adj_flag == MOVING_WALL,
                                   opp_f_in+2.0* 3.0* weights[f]* rho_wall* wp.dot(float_directions[f], u_wall),
                                   new_f)
            
            new_f = wp.where(adj_flag == EQUILIBRIUM,
                                   BGK_feq(weights[f], rho_wall, u_wall, float_directions[f]),
                                   new_f)
            
            # Relaxation Of Equlibrium
            new_f = wp.where(adj_flag == EQUILIBRIUM and is_relax,
                                   sigma*new_f + (1.0 - sigma) *adj_f_in,
                                   new_f)

            f_values[f] = new_f
        rho, u = calc_rho_and_u_vec(f_values, global_id)

        for f in range(num_distributions):
            feq = BGK_feq(weights[f], rho, u, float_directions[f])
            f_out[f, global_id] = f_values[f] - inv_tau * (f_values[f] - feq)

    return fused_kernel

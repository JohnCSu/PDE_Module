import warp as wp
from warp.types import matrix,vector
from ..utils import create_rho_and_u_func
def create_calculate_rho_and_u(num_distributions,float_velocity_directions,dimension,float_dtype):

    rho_and_u = create_rho_and_u_func(False,num_distributions,float_velocity_directions)
    
    @wp.kernel
    def calculate_rho_and_u_kernel(f_in:wp.array2d(dtype=float_dtype),rho:wp.array2d(dtype =float_dtype),u:wp.array2d(dtype =float_dtype)):
        tid = wp.tid()
        rho_v,u_v = rho_and_u(f_in,tid)
        
        rho_v[0,tid] = rho
        
        for i in range(dimension):
            u_v[i,tid] = u[i]
    
    return calculate_rho_and_u_kernel
        
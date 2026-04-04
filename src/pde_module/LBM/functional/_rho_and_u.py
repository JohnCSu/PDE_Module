import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel
def calculate_rho_and_u(kernel:wp_Kernel,f_in:wp_Array,rho:wp_Array,u:wp_Array,device = None):
        wp.launch(kernel,dim = rho.size,inputs = [f_in],outputs = [rho,u],device= device)
        return rho,u
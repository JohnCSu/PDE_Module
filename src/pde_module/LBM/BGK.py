import numpy as np
import warp as wp
from warp.types import vector,matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c,global_to_ijk_c
from .utils import array_to_vec_or_mat

class BGK_collision(LBM_Stencil):
    
    def __init__(self,latticeModel:LatticeModel,grid_shape:tuple[int]):
        super().__init__(latticeModel,grid_shape)
    
    def __call__(self, f_in,tau):
        return super().__call__(f_in,tau)

    @setup
    def initialise(self,f_in,tau):
        self.latticeModel.to_warp()
        self.kernel = create_BGK_collision(self.latticeModel,self.num_distributions,self.dimension,self.grid_shape,self.latticeModel.float_dtype)
        self.f_out = self.create_output_array(f_in)
        
    def forward(self, f_in,tau):
        wp.launch(self.kernel,dim = f_in.shape[-1],inputs=[f_in,1./tau],outputs= [self.f_out])
        return self.f_out
        

def create_BGK_collision(latticeModel:LatticeModel,num_distributions,dimension,grid_shape,float_dtype):
    latticeModel.to_warp()
    float_velocity_directions = latticeModel.float_directions
    weights = latticeModel.weights 
    
    grid_shape = wp.vec3i(grid_shape)
    
    @wp.func
    def BGK_feq(weight:float_dtype,rho:float_dtype,u:vector(dimension,float_dtype),ei:vector(dimension,float_dtype)):
        ei_dot_u = wp.dot(ei,u)
        return weight*rho*(1. + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*wp.dot(u,u))
    
    
    @wp.kernel
    def BGK_collision_kernel(f_in:wp.array2d(dtype =float_dtype),
                             inv_tau:float_dtype,
                             f_out:wp.array2d(dtype =float_dtype)):
        global_id = wp.tid()

        rho = float_dtype(0.0)
        u = vector(float_dtype(0.),length = dimension)
        for f in range(num_distributions):
            rho += f_in[f,global_id]
            u += f_in[f,global_id]*float_velocity_directions[f]
        u /= rho
    
        for f in range(num_distributions):
            feq = BGK_feq(weights[f],rho,u,float_velocity_directions[f])
            f_out[f,global_id] = f_in[f, global_id] - inv_tau * (f_in[f,global_id] - feq)
    
    return BGK_collision_kernel
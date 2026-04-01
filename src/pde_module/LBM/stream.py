import numpy as np
import warp as wp
from warp.types import vector
from pde_module.utils.types import wp_Array,wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c,xijk_to_global_c
from .utils import get_adjacent_ijk

class Streaming(LBM_Stencil):
    grid_shape:tuple[int]
    def __init__(self,latticeModel:LatticeModel,grid_shape:tuple[int]):
        super().__init__(latticeModel,grid_shape)
        
    def __call__(self, f_in:wp_Array):
        return super().__call__(f_in)
    @setup
    def initialise(self,f_in:wp_Array):
        assert f_in.shape[0] == self.num_distributions
        self.latticeModel.to_warp()
        self.kernel = create_streaming_kernel(f_in,self.latticeModel.int_directions,self.num_distributions,self.grid_shape,self.dimension)
        self.f_out = self.create_output_array(f_in)
        
    def forward(self,f_in:wp_Array) -> wp_Array:
        wp.launch(kernel = self.kernel,dim = self.grid_shape,inputs=[f_in],outputs = [self.f_out])
        return self.f_out
        
        
def create_streaming_kernel(f_arr:wp_Array,int_directions:wp_Matrix,num_distributions:int,grid_shape:tuple[int],dimension:int):
    grid_shape = wp.vec3i(grid_shape)
    
    adjacent_ijk = get_adjacent_ijk(dimension,grid_shape)
    
    @wp.kernel
    def streaming_kernel(f_in:wp.array2d(dtype =f_arr.dtype),
                         f_out:wp.array2d(dtype =f_arr.dtype)):
        i,j,k = wp.tid()
        
        global_id = ijk_to_global_c(i,j,k,grid_shape[0],grid_shape[1],grid_shape[2])
        for f in range(num_distributions):
            vel_dir = -int_directions[f] # N,D so vel_dir is a vector of length D dtype int32
            ni,nj,nk = adjacent_ijk(i,j,k,vel_dir)
            adj_id  = ijk_to_global_c(ni,nj,nk,grid_shape[0],grid_shape[1],grid_shape[2])
            f_out[f,global_id]=  f_in[f,adj_id]
            
    return streaming_kernel



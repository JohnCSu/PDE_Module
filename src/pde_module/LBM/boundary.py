import numpy as np
import warp as wp
from warp.types import vector,matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c


PASS = 0
SOLID_WALL = 1
EQUILIBRIUM = 2


class Boundary(LBM_Stencil):
    '''
    Implements Full-Step Bounceback BC i.e. applies BC Strictly AFTER Streaming
    '''
    
    def __init__(self, latticeModel, grid_shape):
        super().__init__(latticeModel, grid_shape)    
        self.flags = np.zeros(self.grid_shape,dtype=np.uint8)
        
    @setup
    def initialize(self,f_in,u_wall):
        self.latticeModel.to_warp()
        self.indices = np.nonzero(self.flags.ravel())[0].astype(wp.dtype_to_numpy(self.latticeModel.int_dtype))
        self.warp_indices = wp.array(self.indices)
        self.warp_flags = wp.array(self.flags.ravel()[self.indices],dtype= wp.uint8)
        
        self.kernel = create_boundary_kernel(self.num_distributions,
                                             self.latticeModel.opposite_indices,
                                             self.latticeModel.float_directions,
                                             self.latticeModel.weights,
                                             self.dimension,
                                             self.grid_shape,
                                             self.latticeModel.float_dtype)
        
        self.f_out = self.create_output_array(f_in)
    
    def forward(self,f_in,u_wall):
        wp.copy(self.f_out,f_in)
        wp.launch(self.kernel,len(self.indices),[f_in,self.warp_flags,self.warp_indices,u_wall],[self.f_out])
        return self.f_out

def create_boundary_kernel(num_distributions,opposite_index,float_directions,weights,dimension,grid_shape,float_dtype):
    @wp.kernel
    def apply_BC(f_in:wp.array2d(dtype =float_dtype),
                 flags:wp.array1d(dtype = wp.uint8),
                 global_ids:wp.array1d(dtype = int),
                 u_wall:vector(dimension,float_dtype),
                 f_out:wp.array2d(dtype =float_dtype),
                 ):
        tid = wp.tid()
        global_id = global_ids[tid]
        
        bounceback = vector(float_dtype(0.),length = num_distributions)
        
        rho = f_in.dtype(0.0)
        for f in range(num_distributions):
            rho += f_in[f,global_id]
            
        for f in range(num_distributions):
            opp_dir = opposite_index[f]
            bounceback[f] = f_in[opp_dir,global_id] # - f_in.dtype(flags[global_id] == wp.uint8(2))*2.*3.*weights[f]*rho*wp.dot(float_directions[f],u_wall)
            if flags[tid] == wp.uint8(2):
                # wp.printf('%d %d\n',flags[tid],tid)
                bounceback[f] += 2.*3.*weights[f]*rho*wp.dot(float_directions[f],u_wall)
        for f in range(num_distributions):
            f_out[f,global_id] = bounceback[f]
        
    # return apply_BC
    # @wp.kernel
    # def apply_BC(f_in:wp.array2d(dtype =float_dtype),
    #              flags:wp.array1d(dtype = wp.uint8),
    #              global_ids:wp.array1d(dtype = int),
    #              u_wall:vector(dimension,float_dtype),
    #              f_out:wp.array2d(dtype =float_dtype),
    #              ):
    #     tid = wp.tid()
    #     global_id = global_ids[tid]
        
    #     i,j,k = global_to_ijk_c(global_id)
        
    #     bounceback = vector(float_dtype(0.),length = num_distributions)
        
    #     rho = f_in.dtype(0.0)
    #     for f in range(num_distributions):
    #         rho += f_in[f,global_id]
            
            
    #     for f in range(num_distributions):
    #         opp_f = opposite_index[f]
    #         bounceback[f] = f_in[opp_f,global_id]
            
    #         if flags[tid] == wp.uint8(2):
    #             # wp.printf('%d %d\n',flags[tid],tid)
    #             bounceback[f] += 2.*3.*weights[f]*rho*wp.dot(float_directions[f],u_wall)
    #     for f in range(num_distributions):
    #         f_out[f,global_id] = bounceback[f]
        
    return apply_BC




@wp.func
def global_to_ijk_c(global_id:int,Nx:int,Ny:int,Nz:int):
    # How many 2D planes (Nj * Nk) fit into the global_id?
    i = global_id // (Ny * Nz)
    # What's left over after removing those planes?
    remainder = global_id % (Ny * Nz)
    # Within that plane, how many rows (Nk) fit?
    j = remainder // Nz
    # What's left over is the position in the current row
    k = remainder % Nz
    return i,j,k
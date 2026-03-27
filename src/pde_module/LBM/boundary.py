import numpy as np
import warp as wp
from warp.types import vector,matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c,global_to_ijk_c
from .utils import get_adjacent_ijk

FLUID = 0
SOLID_WALL = 1
MOVING_WALL = 2
EQUILIBRIUM = 3

class Boundary(LBM_Stencil):
    '''
    Implements Full-Step Bounceback BC i.e. applies BC Strictly AFTER Streaming
    '''
    
    def __init__(self, latticeModel, grid_shape):
        super().__init__(latticeModel, grid_shape)    
        self.flags = np.zeros(self.grid_shape,dtype=np.uint8)
        self.BC_velocity = np.zeros(self.grid_shape + (self.dimension,),dtype= self.latticeModel.float_dtype)
        self.BC_density = np.ones(self.grid_shape + (self.dimension,),dtype= self.latticeModel.float_dtype)
        
        
        self.groups ={
            '-X':(0,slice(None),slice(None)),
            '+X':(-1,slice(None),slice(None)),
            
            '-Y':(slice(None),0,slice(None)),
            '+Y':(slice(None),-1,slice(None)),
            
            '-Z':(slice(None),slice(None),0),
            '+Z':(slice(None),slice(None),-1),
            
        }
        

    def set_BC(self,ids:str | tuple[np.ndarray | int | slice],boundary_type,velocity_value = 0., density_value = 1.):
        
        match ids:
            case str():
                assert ids in self.groups.keys()
                ids = self.groups[ids]
            case tuple():
                assert len(ids) == 3
                assert all(isinstance(obj,(np.ndarray,int,slice)) for obj in ids)
            case _:
                raise TypeError('Strings or tuples of ndarrays are allowed')
                    
        self.flags[*ids] = boundary_type
        self.BC_velocity[*ids,:] = velocity_value
        self.BC_density[ids] = density_value
        
    
    
    @setup
    def initialize(self,f_in):
        self.latticeModel.to_warp()
        self.indices = np.nonzero(self.flags.ravel())[0].astype(wp.dtype_to_numpy(self.latticeModel.int_dtype))
        self.warp_indices = wp.array(self.indices)
        self.warp_flags = wp.array(self.flags,dtype= wp.uint8)
        
        self.warp_BC_velocity = wp.array(self.BC_velocity.reshape((-1,self.dimension))[self.indices,:],dtype = vector(self.dimension,self.latticeModel.float_dtype))
        self.warp_BC_density = wp.array(self.BC_density.ravel()[self.indices])
        
        
        
        self.kernel = create_boundary_kernel(self.num_distributions,
                                             self.latticeModel.opposite_indices,
                                             self.latticeModel.int_directions,
                                             self.latticeModel.float_directions,
                                             self.latticeModel.weights,
                                             self.dimension,
                                             self.grid_shape,
                                             self.latticeModel.float_dtype)
        
        self.f_out = self.create_output_array(f_in)
    
    def forward(self,f_in):
        wp.copy(self.f_out,f_in)
        wp.launch(self.kernel,len(self.indices),[f_in,self.warp_flags,self.warp_indices
                                                 ,self.warp_BC_velocity,self.warp_BC_density
                                                 ]
                  ,
                  outputs = [self.f_out])
        return self.f_out

def create_boundary_kernel(num_distributions,opposite_index,int_directions,float_directions,weights,dimension,grid_shape,float_dtype):
    grid_shape = wp.vec3i(grid_shape)
    adjacent_ijk = get_adjacent_ijk(dimension,grid_shape)
    
    @wp.kernel
    def apply_BC(f_in:wp.array2d(dtype =float_dtype),
                 flags:wp.array3d(dtype = wp.uint8),
                 global_ids:wp.array1d(dtype = int),
                 u_BCs:wp.array1d(dtype = vector(dimension,float_dtype)),
                 rho_BCs:wp.array1d(dtype = float_dtype),
                 f_out:wp.array2d(dtype =float_dtype),
                 ):
        tid = wp.tid()
        global_id = global_ids[tid]
        u_wall = u_BCs[tid]
        rho_wall = rho_BCs[tid]
        i,j,k = global_to_ijk_c(global_id,grid_shape[0],grid_shape[1],grid_shape[2])
        
        for f in range(num_distributions):
            opp_f = opposite_index[f]
            opp_vel = int_directions[opp_f]
            ni,nj,nk = adjacent_ijk(i,j,k,opp_vel)
            # if : # adj node is fluid
            adj_is_fluid = (flags[ni,nj,nk] == FLUID)
            curr_is_moving = (flags[i,j,k] == MOVING_WALL)
            
            f_out[opp_f,global_id] = f_in.dtype(adj_is_fluid)*f_in[f,global_id] + f_in.dtype(not adj_is_fluid)*f_in[opp_f,global_id]
            f_out[opp_f,global_id] -= f_in.dtype(adj_is_fluid and curr_is_moving)*2.*3.*weights[f]*rho_wall*wp.dot(float_directions[f],u_wall) 
            # # The Above is equivalent to below (avoids if statements)            
            # if adj_is_fluid:
            #     f_out[opp_f,global_id] = f_in[f,global_id]
            #     if flags[i,j,k] == MOVING_WALL: # current is at a Moving Wall
            #         f_out[opp_f,global_id] -= 2.*3.*weights[f]*rho*wp.dot(float_directions[f],u_wall)
                
                
                
        
    return apply_BC





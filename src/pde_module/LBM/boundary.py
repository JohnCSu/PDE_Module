import numpy as np
import warp as wp
from warp.types import vector,matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c,global_to_ijk_c
from .utils import get_adjacent_ijk
from .BGK import create_BGK_feq
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
        
        
        self.BC_velocity = np.full(self.grid_shape + (self.dimension,),np.nan,dtype= self.latticeModel.float_dtype)
        self.BC_density = np.full(self.grid_shape,np.nan,dtype= self.latticeModel.float_dtype)
        
        
        self.groups ={
            '-X':(0,slice(None),slice(None)),
            '+X':(-1,slice(None),slice(None)),
            
            '-Y':(slice(None),0,slice(None)),
            '+Y':(slice(None),-1,slice(None)),
            
            '-Z':(slice(None),slice(None),0),
            '+Z':(slice(None),slice(None),-1),
            
        }
        

    def set_BC(self,ids:str | tuple[np.ndarray | int | slice],boundary_type,velocity = None, density = None):
        
        if boundary_type == 2 or boundary_type == 3:
            assert (velocity is not None or density is not None)
            
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
        if velocity is not None:
            self.BC_velocity[*ids,:] = velocity
        if density is not None:
            self.BC_density[ids] = density
        
    
    
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
    BGK_feq = create_BGK_feq(dimension,float_dtype)
    @wp.func
    def get_adjacent_fluid(i:int,
                        j:int,
                        k:int,
                        flags:wp.array3d(dtype = wp.uint8)):
        idx_vec = vector(0,length = dimension)
        shift = wp.vec2i(-1,1)
        for ii in range(dimension): # Axis
            for jj in range(2):
                idx_vec[ii] = shift[jj]
                ni,nj,nk= adjacent_ijk(i,j,k,idx_vec)
                if flags[ni,nj,nk] == FLUID:
                    # wp.printf('FLUID\n')
                    return ni,nj,nk
                idx_vec[ii] = 0 
                
        return i,j,k
    
    
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
        
        fluid_i,fluid_j,fluid_k = get_adjacent_fluid(i,j,k,flags)
        fluid_adj = ijk_to_global_c(fluid_i,fluid_j,fluid_k,grid_shape[0],grid_shape[1],grid_shape[2])
        u = vector(float_dtype(0.),length = dimension) # Boundary u to use if u not assigned
        # u = wp.vec2f()
        rho = float_dtype(0.) # Rho to use if rho not assigned
        for f in range(num_distributions):
            rho += f_in[f,fluid_adj]
            u += f_in[f,fluid_adj]*float_directions[f]
        u /= rho
        
        
        if wp.isnan(u_wall[0]):
            u_wall = u
        
        if wp.isnan(rho_wall):
            rho_wall = rho
        
        # if j == (grid_shape[1] - 1):
        #     wp.printf('%f:.1f %d \n',rho_wall,j)
            
        # if flags[i,j,k] != EQUILIBRIUM:
        for f in range(num_distributions):
            opp_f = opposite_index[f]
            opp_vel = int_directions[opp_f]
            ni,nj,nk = adjacent_ijk(i,j,k,opp_vel)
            adj_is_fluid = (flags[ni,nj,nk] == FLUID)
            bc_is_moving = (flags[i,j,k] == MOVING_WALL)
            bc_is_equil = (flags[i,j,k] == EQUILIBRIUM)

            
            new_opp_f = wp.where(flags[ni,nj,nk] == FLUID,f_in[f,global_id],f_in[opp_f,global_id]) # if opp dir is Fluiod Do Bounceback False basically do nothing
            new_opp_f -= wp.where(adj_is_fluid and bc_is_moving,2.*3.*weights[f]*rho_wall*wp.dot(float_directions[f],u_wall),float_dtype(0.)) # If moving wall as well add forcing term
            new_opp_f = wp.where(bc_is_equil,BGK_feq(weights[opp_f],rho_wall,u_wall,float_directions[opp_f]),new_opp_f) # If Equil BC, replace with equilibrium
            f_out[opp_f,global_id] = new_opp_f
        # else:
        #     for f in range(num_distributions):
        #         f_out[f,global_id] = 
                # f_out[f,global_id] = f_out[f,fluid_adj]
                
            # # The Above is equivalent to below (avoids if statements)            
            # if adj_is_fluid:
            #     f_out[opp_f,global_id] = f_in[f,global_id]
            #     if flags[i,j,k] == MOVING_WALL: # current is at a Moving Wall
            #         f_out[opp_f,global_id] -= 2.*3.*weights[f]*rho*wp.dot(float_directions[f],u_wall)

            
                
                
        
    return apply_BC




            
        
    





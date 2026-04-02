import numpy as np
import warp as wp
from warp.types import vector,matrix
from pde_module.utils.types import wp_Array,wp_Vector,wp_Matrix,Any
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.LBM_Stencil import LBM_Stencil
from pde_module.utils import ijk_to_global_c,global_to_ijk_c
from .utils import get_adjacent_ijk,create_rho_and_u_func
from .BGK import create_BGK_feq
from math import prod
FLUID = 0
SOLID_WALL = 1
MOVING_WALL = 2
EQUILIBRIUM = 3


class FusedLBMKernel(LBM_Stencil):
    '''
    Fuses the Stream, BC and BGK collision operator into one single kernel
    '''
    @classmethod
    def from_LBM_Mesh(cls, mesh):
        return super().from_LBM_Mesh(mesh,'flags','groups')
    
    def __init__(self, latticeModel, grid_shape,flags,groups):
        super().__init__(latticeModel, grid_shape)    
        self.flags = flags
        self.BC_velocity = np.full(self.grid_shape + (self.dimension,),0,dtype= self.latticeModel.float_dtype)
        self.BC_density = np.full(self.grid_shape,1,dtype= self.latticeModel.float_dtype)
        self.groups = groups
        
        
    def set_BC(self,ids:str | tuple[np.ndarray | int | slice],boundary_type,velocity = None, density = None):
        '''
        Set Boundary Condition Values for node Ids or string.
        
        If density is set to a negative value, then damping is added 
        
        '''
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
    def initialize(self,f_in,tau,f_out = None):
        self.latticeModel.to_warp()
        
        self.warp_flags = wp.array(self.flags,dtype= wp.uint8)
        self.warp_BC_velocity = wp.array(self.BC_velocity,dtype = vector(self.dimension,self.latticeModel.float_dtype))
        self.warp_BC_density = wp.array(self.BC_density)
        
        self.kernel = create_fusedLBMKernel(
            self.latticeModel.weights,
            self.latticeModel.opposite_indices,
            self.latticeModel.int_directions,
            self.latticeModel.float_directions,
            self.latticeModel.num_distributions,
            self.grid_shape,
            self.dimension,
            self.latticeModel.float_dtype
        ) 
        
        self.f_out = self.create_output_array(f_in)
        self.num_nodes = prod(self.grid_shape)
    
    def forward(self,f_in,tau,f_out):
        output_array = f_out if f_out is not None else self.f_out
        wp.launch(self.kernel,dim = self.num_nodes,inputs=[
            f_in,
            1/tau,
            self.warp_flags,
            self.warp_BC_velocity,
            self.warp_BC_density,
        ],
                  outputs=[
                      output_array
                  ])
        
        return output_array
        
def create_fusedLBMKernel(
                          weights:wp_Vector,
                          opposite_index:wp_Vector,
                          int_directions:wp_Matrix,
                          float_directions:wp_Matrix,
                          num_distributions:int,
                          grid_shape:tuple[int],
                          dimension:int,
                          float_dtype):
    
    grid_shape = wp.vec3i(grid_shape)
    adjacent_ijk = get_adjacent_ijk(dimension,grid_shape)
    calc_rho_and_u = create_rho_and_u_func(True,num_distributions,float_directions,dimension,float_dtype)
    BGK_feq = create_BGK_feq(dimension,float_dtype)
    @wp.kernel
    def fused_kernel(f_in:wp.array2d(dtype =float_dtype),
                     inv_tau:float_dtype,
                     flags:wp.array3d(dtype = wp.uint8),
                     u_BC:wp.array3d(dtype = vector(dimension,float_dtype)),
                     rho_BC:wp.array3d(dtype = float_dtype),
                     f_out:wp.array2d(dtype =float_dtype)):
        tid = wp.tid()
        global_id = tid
        i,j,k = global_to_ijk_c(global_id,grid_shape[0],grid_shape[1],grid_shape[2])
        f_values = vector(float_dtype(0.),length = num_distributions)
        
        for f in range(num_distributions):
            vel_dir = -int_directions[f] # N,D so vel_dir is a vector of length D dtype int32
            ni,nj,nk = adjacent_ijk(i,j,k,vel_dir)
            adj_id  = ijk_to_global_c(ni,nj,nk,grid_shape[0],grid_shape[1],grid_shape[2])
            opp_f = opposite_index[f]
            
            rho_wall = rho_BC[ni,nj,nk]
            u_wall = u_BC[ni,nj,nk]
            adj_flag = flags[ni,nj,nk]
            adj_f_in = f_in[f,adj_id]
            opp_f_in = f_in[opp_f,global_id]
            
            f_values[f] = wp.where(adj_flag == FLUID,adj_f_in,f_values[f])
            f_values[f] = wp.where(adj_flag == SOLID_WALL,opp_f_in,f_values[f])
            f_values[f] = wp.where(adj_flag == MOVING_WALL,opp_f_in + 2.*3.*weights[f]*rho_wall*wp.dot(float_directions[f],u_wall),f_values[f])
                        
            # # Stream if fluid
            # if adj_flag == FLUID:
            #     f_values[f]=  f_in[f,adj_id]
            #     # f_values[f]= f_in[opp_f,global_id]
            # # Bounceback if Solidwall
            # elif flags[ni,nj,nk] == SOLID_WALL:
            #     f_values[f]= f_in[opp_f,global_id]
            # elif flags[ni,nj,nk] == MOVING_WALL:
            # # # Bounceback if Solidwall + mom if Moving Wall
            #     f_values[f]= f_in[opp_f,global_id] + 2.*3.*weights[f]*rho_wall*wp.dot(float_directions[f],u_wall)
        
        rho,u = calc_rho_and_u(f_values,global_id)
        
        #Collision Operator
        for f in range(num_distributions):
            # f_out[f,global_id] = f_values[f]
            feq = BGK_feq(weights[f],rho,u,float_directions[f])
            f_out[f,global_id] = f_values[f] - inv_tau * (f_values[f] - feq)
        
    return fused_kernel
        
        
        

            
        
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from .boundary import Boundary
import warp as wp
from warp.types import vector,matrix,type_is_vector
from ..Stencil.hooks import *
import numpy as np
from ..stencil_utils import create_stencil_op,eligible_dims_and_shift

class ImmersedBoundary(Boundary):
    '''
    Module for objects inside the domain. Only First and Second order approximations availiable
    '''
    # def __init__(self,field,dx,ghost_cells:int):
    def __init__(self,field,dx,ghost_cells:int):
        super().__init__(field,dx,ghost_cells)
        self.bitmask = np.zeros(field.shape,dtype = np.int8)
        
    
    def from_bool_func(self,fn,meshgrid):
        '''
        Provide a func such that False is given for values outside and True for values inside the boundary    
        e.g.
        f(x,y) -> sqrt(x**2+y**2) < R
        '''
        # We have a 3,grid_shape array
        assert meshgrid[0].shape == self.grid_shape
        bitmask = fn(*meshgrid)
        self.bitmask += bitmask.astype(np.int8)
        
    def finalize(self):
        '''
        calculate the final boundaries of the bitmask from multiple different shapes
        '''
        self._find_solid_boundary()
        self._find_fluid_boundary()
        self._find_fluid_neighbors()
        self.define_boundary_value_and_type_arrays(self.solid_boundary)
        
    def _find_solid_boundary(self):
        # From the bitmask we need to go through all points and find the boundary
        self.solid_indices = np.stack(np.nonzero(self.bitmask),axis= -1,dtype=np.int32)
        is_solid_boundary = np.zeros(shape = len(self.solid_indices),dtype= np.bool) 
        # Find solid Boundary
        locate_boundary_kernel = locate_boundary(self.grid_shape,self.ghost_cells)
        wp.launch(locate_boundary_kernel,len(self.solid_indices),inputs=[self.bitmask,self.solid_indices],outputs=[is_solid_boundary],device='cpu')
        self.solid_boundary = self.solid_indices[is_solid_boundary]
        self.interior_solids= self.solid_indices[~is_solid_boundary]
        # Convert to flattened indices
        
        L,H,W = self.grid_shape
        x,y,z = self.interior_solids[:,0],self.interior_solids[:,1],self.interior_solids[:,2]
        self.flat_array = (x*H*W) + y*W + z
        
        
    def _find_fluid_boundary(self):
        #Find Fluid Boundary
        identify_fluid_boundary = create_identify_fluid_boundary_kernel(self.grid_shape,self.ghost_cells)
        max_neighbors = self.dimension*2
        
        fluid_boundary = wp.empty((len(self.solid_boundary),max_neighbors),dtype=wp.vec3i,device='cpu')
        fluid_boundary.fill_(wp.vec3i(-1,-1,-1))
        
        wp.launch(identify_fluid_boundary,dim = len(self.solid_boundary),inputs=[self.bitmask,self.solid_boundary],outputs=[fluid_boundary],device= 'cpu')
        fluid_boundary = fluid_boundary.flatten().numpy()
        
        self.fluid_boundary = np.unique(fluid_boundary[fluid_boundary[:,0] != -1],axis= 0)
    
    def _find_fluid_neighbors(self):    
        # identify Neighbors of fluid
        dim = self.dimension
        self.fluid_neighbors = np.zeros(shape = (len(self.solid_boundary),dim,2),dtype=np.int8)
        identify_neighbors_kernel = identify_fluid_neighbors(self.grid_shape,self.ghost_cells)
        wp.launch(identify_neighbors_kernel,dim = len(self.solid_boundary), inputs = [self.bitmask,self.solid_boundary],outputs=[self.fluid_neighbors],device = 'cpu')
    
    
    @setup
    def to_warp(self,*args,**kwargs):
        adj_matrix = matrix((self.dimension,2),dtype = wp.int8)
        self.warp_solid_boundary_indices = wp.array(self.solid_boundary,dtype=wp.vec3i)
        self.warp_fluid_neighbors = wp.array(self.fluid_neighbors,dtype=adj_matrix)
        self.warp_boundary_type =wp.array(self.boundary_type)
        self.warp_boundary_value = wp.array(self.boundary_value,dtype = self.input_dtype)
        
        
        self.warp_interior_solids = wp.array(self.interior_solids,dtype=wp.vec3i)
        self.warp_interior_solid_indices = [wp.array(arr,dtype = int) for arr in np.moveaxis(self.interior_solids,0,-1)]
        
    @setup
    def initialize_kernel(self, input_array, *args, **kwargs):
        self.kernel = create_staircase_boundary_kernel(input_array.dtype,self.grid_shape,self.ghost_cells,self.dx)
        self.fill_solids_inplace_kernel = create_fill_solids_inplace(input_array.dtype)
    
    @before_forward
    def copy_array(self,input_array,*args,**kwargs):
        wp.copy(self.output_array,input_array)
    
    
    def forward(self, input_array,fill_value = 0.,*args,**kwargs):
        wp.launch(self.kernel,dim = (len(self.warp_solid_boundary_indices),self.inputs[0]),inputs= [input_array,
                                                                     self.warp_solid_boundary_indices,
                                                                     self.warp_boundary_value,
                                                                     self.warp_boundary_type,
                                                                     self.warp_fluid_neighbors
                                                                     ],
                  outputs= [self.output_array])
        
        wp.launch(self.fill_solids_inplace_kernel,dim = len(self.warp_interior_solids),inputs=[self.warp_interior_solids,fill_value],outputs= [self.output_array])
        return self.output_array 



def create_fill_solids_inplace(input_dtype): 
    
    @wp.kernel
    def fill_solids_inplace(
        solid_boundary_indices:wp.array(dtype=wp.vec3i),
        value: float,
        new_values:wp.array3d(dtype = input_dtype),
        
    ):
        tid = wp.tid() #Boundary Index
        solidID = solid_boundary_indices[tid]
        x = solidID[0]
        y = solidID[1]
        z = solidID[2]
        new_values[x,y,z] = input_dtype(value)
    return fill_solids_inplace


def create_staircase_boundary_kernel(input_dtype,grid_shape,ghost_cells,dx):
    '''
    For Staircase appoximation:
    dirichlet = set solid bound to the value
    vonneumann -> average from different directions
    
    '''
    
    DIRICHLET = wp.int8(1)
    VON_NEUMANN = wp.int8(2)
    
    FLUID_CELL = wp.int8(0)
    SOLID_CELL = wp.int8(1)
    
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    
    dim = len(eligible_dims)
    adj_matrix = matrix((dim,2),dtype = wp.int8)
    
    float_type = input_dtype._wp_scalar_type_
    dx = float_type(dx)
    shift = wp.vec2i(-1,1)
    @wp.kernel
    def boundary_kernel(
        current_values:wp.array3d(dtype = input_dtype),
        solid_boundary_indices:wp.array(dtype=wp.vec3i),
        boundary_value:wp.array(dtype = input_dtype),
        boundary_type:wp.array2d(dtype = wp.int8),
        neighbors:wp.array(dtype=adj_matrix),
        new_values:wp.array3d(dtype = input_dtype),
        ):
        
        tid,var = wp.tid() #Boundary Index
        
        solidID = solid_boundary_indices[tid]
        x = solidID[0]
        y = solidID[1]
        z = solidID[2]
        BC_val = boundary_value[tid][var]
        if boundary_type[tid][var] == DIRICHLET:
                        new_values[x,y,z][var] =  BC_val
        elif boundary_type[tid][var] == VON_NEUMANN:
                        # Find Fluid Neighbors 
                        neighbor_mat = neighbors[tid]
                        avg_value = float_type(0.)
                        n = float_type(1.)
                        for axis in range(dim):
                            j = eligible_dims[axis]
                            adj_vec = wp.vec3i()
                            for i in range(2):
                                if neighbor_mat[axis,i] == FLUID_CELL:
                                    
                                    adj_vec[j] = shift[i] # -1 or 1
                                    
                                    fluid_idx = solidID + adj_vec
                                    contribution_val =  current_values[fluid_idx[0],fluid_idx[1],fluid_idx[2]][var]- float_type(shift[i])*BC_val*dx 
                                    # Add to running average
                                    avg_value += (contribution_val - avg_value)/n
                                    n += float_type(1.)
                        
                        new_values[x,y,z][var] = avg_value
                                
    return boundary_kernel
        





def create_identify_fluid_boundary_kernel(grid_shape,ghost_cells):
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    
    dim = len(eligible_dims)
    
    # fluid_indices have shape len(soldi_idx),dim
    
    shift = wp.vec2i(-1,1)
    @wp.kernel
    def identify_fluid_boundary(
                        bit_array:wp.array3d(dtype=wp.int8),
                        solid_indices:wp.array(dtype = wp.vec3i),
                        fluid_indices: wp.array2d(dtype=wp.vec3i)                 
                        ):
        tid = wp.tid()
        
        boundary_idx = solid_indices[tid]
        
        for d in range(dim):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            # adj_vec[j] = 1
            for i in range(2):
                adj_vec[j] = shift[i]
                adj = boundary_idx + adj_vec
                if bit_array[adj[0],adj[1],adj[2]] == 0:
                    fluid_indices[tid,i+j*2] = adj
            
            
    return identify_fluid_boundary



def identify_fluid_neighbors(grid_shape,ghost_cells):
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    
    dim = len(eligible_dims)
    adj_matrix = matrix((dim,2),dtype = wp.int8)
    
    @wp.kernel
    def identify_neighbors_kernel(
                        bit_array:wp.array3d(dtype=wp.int8),
                        solid_indices:wp.array(dtype = wp.vec3i),
                        adjacent: wp.array(dtype=adj_matrix)                 
                        ):
        tid = wp.tid()
        
        boundary_idx = solid_indices[tid]
        mat = adj_matrix()
        
        for d in range(wp.static(len(eligible_dims))):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            adj_vec[j] = 1
            
            adj_l = boundary_idx - adj_vec
            adj_r = boundary_idx + adj_vec
            
            mat[d,0] = bit_array[adj_l[0],adj_l[1],adj_l[2]] 
            mat[d,1] = bit_array[adj_r[0],adj_r[1],adj_r[2]]
            
        adjacent[tid] = mat
            
            
    return identify_neighbors_kernel
        
        
        


def locate_boundary(grid_shape,ghost_cells):
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    # print(eligible_dims)
    @wp.kernel
    def locate_boundary_kernel(bit_array:wp.array3d(dtype=wp.int8),
                        solid_indices:wp.array(dtype = wp.vec3i),
                        is_boundary_indices:wp.array(dtype=wp.bool)):
        tid = wp.tid()
        
        solid_index = solid_indices[tid]
        # wp.printf('Solid_index:%d, %d, %d \n',solid_index[0],solid_index[1],solid_index[2])
        for d in range(wp.static(len(eligible_dims))):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            adj_vec[j] = 1
            
            adj_l = solid_index - adj_vec
            adj_r = solid_index + adj_vec
            
            # wp.printf('right :%d, %d, %d %d \n',adj_r[0],adj_r[1],adj_r[2],bit_array[adj_l[0],adj_l[1],adj_l[2]] )
            # wp.printf('left :%d, %d, %d %d \n',adj_l[0],adj_l[1],adj_l[2],bit_array[adj_r[0],adj_r[1],adj_r[2]])
            
            if (bit_array[adj_l[0],adj_l[1],adj_l[2]] == wp.int8(0))  or (bit_array[adj_r[0],adj_r[1],adj_r[2]] == wp.int8(0)):
                is_boundary_indices[tid] = True
                return    
            
    return locate_boundary_kernel
        
        
    
    
    
    
        
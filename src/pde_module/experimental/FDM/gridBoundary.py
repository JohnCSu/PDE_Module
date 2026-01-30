from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from .boundary import Boundary
import warp as wp
from warp.types import vector,matrix,type_is_vector
# from .types import *
from ..Stencil.hooks import *
import numpy as np

class GridBoundary(Boundary):
    # def __init__(self,field,dx,ghost_cells:int):
    def __init__(self,field,dx,ghost_cells:int):
        super().__init__(field,dx,ghost_cells)        
        self.define_boundary_xyz_indices()
        self.define_groups()
        self.define_interior_adjacency()
        self.define_boundary_value_and_type_arrays(self.boundary_xyz_indices)

    def define_boundary_xyz_indices(self,*args,**kwargs):
        
        boundary_xyz_indices = []
        
        for i in range(3):
            if self.grid_shape_without_ghost[i] == 1:
                continue 
            axis_limits = (0,self.grid_shape_without_ghost[i]-1)
            for axis_lim in axis_limits:
                side = list(self.grid_shape_without_ghost)
                side[i] = 1
                
                indices = np.indices(side,dtype = np.int32)
                indices = np.moveaxis(indices,0,-1).reshape(-1,3)
                indices[:,i] = axis_lim
                
                for ax in range(3):
                    if self.grid_shape_without_ghost[ax] != 1 :
                        indices[:,ax] += self.ghost_cells
                # print(indices)
                
                
                boundary_xyz_indices.append(indices)
                
        self.boundary_xyz_indices = np.unique(np.concat(boundary_xyz_indices,axis = 0,dtype=np.int32),axis = 0).astype(np.int32)
        # print()
    
    def define_groups(self,*args,**kwargs):
        '''
        TODO:
        Can be optimized with numba so we only to one sweep through all boundary indices
        '''
        
        self.boundary_indices = np.arange(len(self.boundary_xyz_indices),dtype = np.int32)
        self.groups['ALL'] = self.boundary_indices
        
        for axis,axis_name in enumerate(['X','Y','Z']):
            if self.grid_shape_without_ghost[axis] == 1:
                continue
            coords = self.grid_shape_without_ghost[axis]
            
            axis_limits = [0 + self.ghost_cells,coords-1 + self.ghost_cells] # We have the shifts existing
            
            for axis_limit,parity in zip(axis_limits,['-','+']):
                name = parity + axis_name
                self.groups[name] = self.boundary_indices[self.boundary_xyz_indices[:,axis] == axis_limit]
        
    def define_interior_adjacency(self):
        self.boundary_interior = np.zeros(shape = (len(self.boundary_xyz_indices),3),dtype = np.int32)
        for i,axis in enumerate(['X','Y','Z']):
            if self.grid_shape_without_ghost[i] == 1:
                continue
            for parity in ['-','+']:
                sign = 1 if parity == '-' else -1
                key = parity+axis
                index = self.groups[key]
                self.boundary_interior[index,i] = sign
        
    
    def no_slip(self,group:str|int|np.ndarray|list|tuple):
        '''
        If the input dtype matches the vector length, assumes it is a velocity vector
        '''
        assert self.inputs[0] == self.dimension, 'Valid only when input_dtype is vector with same length equal to dimension of field'
        self.dirichlet_BC(group,0.)
    
    def impermeable(self,group):
        '''
        for side walls set normal velocity to zero
        '''
        assert self.inputs[0] == self.dimension, 'Valid only when input_dtype is vector with same length equal to dimension of field'
        
        assert group in self.groups.keys() and group in {'-X','+X','-Y','+Y','-Z','+Z'}, "{'-X','+X','-Y','+Y','-Z','+Z'} are valid groups"
        
        axis_name = group[-1]
        indices = ['X','Y','Z']
        axis = indices.index(axis_name)
        self.vonNeumann_BC(group,0.,0)
        self.dirichlet_BC(group,0.,axis)    
    
    @setup
    def to_warp(self,*args,**kwargs):
        self.warp_boundary_xyz_indices = wp.array(self.boundary_xyz_indices,dtype=wp.vec3i)
        self.warp_boundary_interior =wp.array(self.boundary_interior,dtype = wp.vec3i)
        self.warp_boundary_type =wp.array(self.boundary_type)
        self.warp_boundary_value = wp.array(self.boundary_value,dtype = self.input_dtype)
        
    @setup
    def initialize_kernel(self, input_array, *args, **kwargs):
        self.kernel = create_boundary_kernel(self.input_dtype,self.ghost_cells,self.dx)
    
    @before_forward
    def copy_array(self,input_array,*args,**kwargs):
        wp.copy(self.output_array,input_array)
        
    def forward(self, input_array, *args, **kwargs):
        # wp.copy(self.output_array,input_array)
        wp.launch(
            kernel=self.kernel,
            dim = (len(self.boundary_indices),*self.input_dtype_shape),
            inputs=[
                input_array,
                self.warp_boundary_xyz_indices,
                self.warp_boundary_value,
                self.warp_boundary_type,
                self.warp_boundary_interior,
            ],
            outputs=[
                self.output_array
            ]
        )
        return self.output_array
    
    
    
    
def create_boundary_kernel(input_dtype,ghost_cells,dx):
    dx = dx
    DIRICHLET = wp.int8(0)
    VON_NEUMANN = wp.int8(1)
    
    float_type = input_dtype._wp_scalar_type_
    @wp.kernel
    def boundary_kernel(
        current_values:wp.array3d(dtype = input_dtype),
        boundary_xyz_indices:wp.array(dtype=wp.vec3i),
        boundary_value:wp.array(dtype = input_dtype),
        boundary_type:wp.array2d(dtype = wp.int8),
        boundary_interior:wp.array(dtype=wp.vec3i),
        new_values:wp.array3d(dtype = input_dtype),
        ):
        
        i,var = wp.tid() #Boundary Index
        
        nodeID = boundary_xyz_indices[i]
        x = nodeID[0]
        y = nodeID[1]
        z = nodeID[2]
        
        # wp.printf('%d,%d,%d,   %d,%d,%d,\n',x,y,z, nodeID[0],nodeID[1],nodeID[2])
        interior_vec = boundary_interior[i]         
        #Update Ghost value
        for axis in range(3):
            if interior_vec[axis] != 0:                    
                inc_vec = wp.vec3i()
                inc_vec[axis] = interior_vec[axis]
                ghostID = nodeID - inc_vec
                adjID = nodeID + inc_vec
                val =boundary_value[i][var]
                if boundary_type[i][var] == DIRICHLET:
                    new_values[x,y,z][var] =  val
                    new_values[ghostID[0],ghostID[1],ghostID[2]][var] =  type(dx)(2.)*val - current_values[adjID[0],adjID[1],adjID[2]][var]
                elif boundary_type[i][var] == VON_NEUMANN:
                    new_values[ghostID[0],ghostID[1],ghostID[2]][var] = val - wp.sign(float_type(inc_vec[axis]))*current_values[adjID[0],adjID[1],adjID[2]][var]*dx
        

                    
            # for j in range(wp.static(ghost_cells)):
    return boundary_kernel
        
    
    
    
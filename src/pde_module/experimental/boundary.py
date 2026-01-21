from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix
# from .types import *
from .hooks import *
import numpy as np

class GridBoundary(ExplicitUniformGridStencil):
    def __init__(self,field,ghost_cells:int,dx):
        
        inputs = self.get_shape_from_dtype(field.dtype)
        super().__init__(inputs ,inputs, dx, field.dtype._wp_scalar_type_)
        assert type(ghost_cells) is int and ghost_cells > 0
        self.ghost_cells = ghost_cells
        
        self.field_shape = self.field_shape_with_no_ghost_cells(field.shape,ghost_cells)
        self.ghost_shape = field.shape

        
        self.define_boundary_xyz_indices()
        self.define_groups()
        self.define_values_and_masks()
        self.define_interior_adjacency()
        
        
    def define_boundary_xyz_indices(self,*args,**kwargs):
        
        boundary_xyz_indices = []
        
        for i in range(3):
            if self.field_shape[i] == 1:
                continue 
            axis_limits = (0,self.field_shape[i]-1)
            for axis_lim in axis_limits:
                side = list(self.field_shape)
                side[i] = 1
                
                indices = np.indices(side,dtype = np.int32)
                indices = np.moveaxis(indices,0,-1).reshape(-1,3)
                indices[:,i] = axis_lim
                
                for ax in range(3):
                    if self.field_shape[ax] != 1 :
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
        
        self.groups = {}
        
        self.boundary_indices = np.arange(len(self.boundary_xyz_indices),dtype = np.int32)
        self.groups['ALL'] = self.boundary_indices
        
        for axis,axis_name in enumerate(['X','Y','Z']):
            if self.field_shape[axis] == 1:
                continue
            coords = self.field_shape[axis]
            
            axis_limits = [0 + self.ghost_cells,coords-1 + self.ghost_cells] # We have the shifts existing
            
            for axis_limit,parity in zip(axis_limits,['-','+']):
                name = parity + axis_name
                self.groups[name] = self.boundary_indices[self.boundary_xyz_indices[:,axis] == axis_limit]
        
    
    def define_values_and_masks(self,*args, **kwargs):
        shape = (len(self.boundary_xyz_indices),) + self.input_dtype_shape
        self.boundary_value = np.zeros(shape,dtype=wp.dtype_to_numpy(self.input_scalar_type))
        self.boundary_type = np.zeros_like(self.boundary_value,dtype = np.int8)

    
    def define_interior_adjacency(self):
        self.boundary_interior = np.zeros(shape = (len(self.boundary_xyz_indices),3),dtype = np.int32)
        for i,axis in enumerate(['X','Y','Z']):
            if self.field_shape[i] == 1:
                continue
            for parity in ['-','+']:
                sign = 1 if parity == '-' else -1
                key = parity+axis
                index = self.groups[key]
                self.boundary_interior[index,i] = sign
        
        
    
    def set_BC(self,face_ids:str|int|np.ndarray|list|tuple,value:float,boundary_type:int,outputs_ids:int|np.ndarray|list|tuple|None):
        if isinstance(face_ids,str):
            assert face_ids in self.groups.keys()
            face_ids = self.groups[face_ids]
            
        assert isinstance(face_ids,(np.ndarray,list,tuple,int))
        
        assert isinstance(outputs_ids,(int,list,tuple,np.ndarray)) or outputs_ids is None
        
        if isinstance(outputs_ids,int):
            assert 0 <= outputs_ids < self.num_inputs
        elif isinstance(outputs_ids,(list,tuple,np.ndarray)):
            output_ids = np.array(output_ids,dtype = np.int32)
            assert np.all( 0 <= output_ids < self.num_inputs)
        
        if outputs_ids is None:
            outputs_ids = slice(None)
        
        assert isinstance(value,float), 'Value must be type float'
        
        self.boundary_type[face_ids,outputs_ids] = boundary_type # For Dirichlet
        self.boundary_value[face_ids,outputs_ids] = value
        
    def dirichlet_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,0,outputs_ids)        
    
    def vonNeumann_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,1,outputs_ids)
        
        
    @setup
    def to_warp(self,*args,**kwargs):
        self.warp_boundary_xyz_indices = wp.array(self.boundary_xyz_indices,dtype=wp.vec3i)
        # self.warp_boundary_indices = wp.array(self.boundary_indices)
        self.warp_boundary_interior =wp.array(self.boundary_interior,dtype = wp.vec3i)
        self.warp_boundary_type =wp.array(self.boundary_type)
        self.warp_boundary_value = wp.array(self.boundary_value,dtype = self.input_dtype)
        
        
    @setup
    def initialize_kernel(self, input_array, *args, **kwargs):
        self.kernel = create_boundary_kernel(self.input_dtype,self.ghost_cells,self.dx)
    
    def forward(self, input_array, *args, **kwargs):
        wp.copy(self.output_array,input_array)
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
        
        # new_values[x,y,z][0] = 1.
        # boundary_type[i][var] = 1.
        
        if boundary_type[i][var] == DIRICHLET:
            new_values[x,y,z][var] =  boundary_value[i][var]
            
            for j in range(3):
                if interior_vec[j] != 0:                    
                    inc_vec = wp.vec3i(0,0,0)
                    inc_vec[j] = interior_vec[j]
                    ghostID = nodeID - inc_vec
                    adjID = nodeID + inc_vec
                    new_values[ghostID[0],ghostID[1],ghostID[2]][var] =  type(dx)(2.)*boundary_value[i][var] - current_values[adjID[0],adjID[1],adjID[2]][var]
                    
                    
            # for j in range(wp.static(ghost_cells)):
    return boundary_kernel
        
        
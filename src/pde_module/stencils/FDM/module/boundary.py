
import numpy as np
import warp as wp
from pde_module.stencils import StencilModule 
from ..kernels.boundary import create_boundary_kernel
from ..functional.boundary import boundary
from pde_module.grids import Grid

class GridBoundary(StencilModule):
    '''Module to define BC at the boundary of grid. Current implementation are time constant BC For immersed boundaries see ... instead'''
    def __init__(self, grid:Grid, num_inputs,*,dynamic_array_alloc = True, **kwargs):
        super().__init__(grid, num_inputs,num_inputs, dynamic_array_alloc, **kwargs)
        
        self.groups = dict()
        
        
        self.kernel = create_boundary_kernel(self.num_inputs)
        self.is_warp = False 
        self.zero_output_array = False

        
        self.boundary_indices = grid.boundary_node_indices    
        self.boundary_type = -1*np.ones(shape = (len(self.boundary_indices),self.num_inputs),dtype=np.int32)
        self.boundary_value = np.zeros_like(self.boundary_type).astype(np.float32)

        self.set_groups()
        self.set_interior_adj_array()
    
    
    def set_groups(self):
        # assert hasattr(self,"boundary_indices")
        for axis,axis_name in enumerate(['X','Y','Z'][:self.dimension]):
            
            coords = self.grid.nodal_shape[axis]
            axis_lim = [0,coords-1]
            
            for axis_side,side in zip(axis_lim,['-','+']):
                name = side + axis_name
                self.groups[name] = np.argwhere(self.boundary_indices[:,axis] == axis_side)
                
        self.groups['ALL'] = np.arange(len(self.boundary_indices),dtype = np.int32)
        
        
    def set_interior_adj_array(self):
        interior_adjacency = np.zeros(shape = (len(self.boundary_indices),3),dtype = np.int32)
        
        axes = ['X','Y','Z'][:self.dimension]
        
        for i,axis in enumerate(axes):
            for sign in ['-','+']:
                mask = self.groups[sign+axis]
                
                inc = 1 if sign == '-' else -1
                
                interior_adjacency[mask,i] = inc
                
        self.interior_adjaceny, self.interior_indices = np.unique(interior_adjacency,axis =0,return_inverse= True)
                
            
            
    def set_BC(self,face_ids:str|int|np.ndarray|list|tuple,value:float,boundary_type:int,outputs_ids:int|np.ndarray|list|tuple|None):
        
        
        if isinstance(face_ids,str):
            assert face_ids in self.groups.keys()
            face_ids = self.groups[face_ids]
            
            
        assert isinstance(face_ids,(np.ndarray,list,tuple,int))
        
        assert isinstance(outputs_ids,(int,list,tuple,np.ndarray)) or outputs_ids is None
        
        if isinstance(outputs_ids,int):
            assert outputs_ids < self.num_inputs and outputs_ids >= 0
        elif isinstance(outputs_ids,(list,tuple,np.ndarray)):
            for output_id in outputs_ids:
                assert isinstance(output_id,int)
                assert 0 <= output_id < self.num_inputs
        
        
        if outputs_ids is None:
            outputs_ids = slice(None)
        
        assert isinstance(value,float), 'Value must be type float'
        
        self.boundary_type[face_ids,outputs_ids] = boundary_type # For Dirichlet
        self.boundary_value[face_ids,outputs_ids] = value
        
    
    def dirichlet_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,0,outputs_ids)        
    
    def vonNeumann_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,1,outputs_ids)
        
        
    def forward(self,input_array):
        wp.copy(dest = self.output_array, src =input_array)
        thread_shape = (len(input_array),len(self.boundary_indices),self.num_inputs)
        return boundary(self.kernel,input_array,self.grid.dx,thread_shape,self.boundary_indices,self.boundary_type,self.boundary_value,self.interior_indices,self.interior_adjaceny,self.grid.levels,self.output_array)

    
    def init_stencil(self,input_array):
        self.init_stencil_flag = False
        self.to_warp()
        self.init_output_array(input_array)
        
    
    def to_warp(self):
        if self.is_warp is False:
            self.is_warp = True
            check_arr = (self.boundary_type == -1)
            if np.any(check_arr):
                raise ValueError(f'boundary ids {np.argwhere(check_arr)} do not have boundary types assigned to them')
            
            self.boundary_type = wp.array(self.boundary_type,dtype=wp.vec(length = self.num_inputs,dtype = int))
            self.boundary_value = wp.array(self.boundary_value,dtype=wp.vec(length = self.num_inputs,dtype = float))
            self.boundary_indices = wp.array(self.boundary_indices,dtype=int)

            
            self.interior_adjaceny = wp.array(self.interior_adjaceny,dtype= wp.vec3i)
            self.interior_indices = wp.array(self.interior_indices,dtype= int)
        
        
        
        
    
    
    
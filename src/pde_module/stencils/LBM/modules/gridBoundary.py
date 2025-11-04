from ....stencil_module import StencilModule
from pde_module.grids.cell_grid import UniformCellGrid
import warp as wp
import numpy as np

class GridBoundary(StencilModule):
    '''
    Base Class for defining Boundary conditions for Cell Grids
    '''
    def __init__(self, grid:UniformCellGrid, num_inputs= 1, num_outputs=1, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, num_inputs, num_outputs, dynamic_array_alloc, float_type)
    
    def define_indices(self):
        '''
        These are essentially cell faces so we 
        '''
        self.groups = {}
        index = []
        axes = ['X','Y','Z'][:self.dimension]
        shift = 0
        for i,axis in enumerate(axes):
            for side in ['-','+']:
                boundary_key = side+axis
                
                boundary_shape = list(self.grid.shape)[:self.dimension]
                # boundary_shape = boundary_shape[:i] + boundary_shape[i+1:]
                boundary_shape[i] = 1
                indices =np.indices(boundary_shape)
                indices = np.moveaxis(indices,0,-1)
                indices[i] = 0 if side == '-' else (self.grid.shape[i] -1)
                
                index.append(indices)
                self.groups[boundary_key] = np.arange(len(indices),dtype = np.int32) + shift
                shift += len(indices)
            
        self.boundary_indices = np.concatenate(index)
        self.boundary_values = np.zeros(shape = len(self.boundary_indices,self.grid.num_discrete_velocities))
        self.boundary_type = np.zeros(shape = len(self.boundary_indices))
    
    
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
        
        
                
            
                
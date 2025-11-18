from ...stencil_module import StencilModule
from pde_module.grids import Grid
import warp as wp
import numpy as np
from pde_module.grids.latticeModels import D2Q9_Model
from ..kernels.GridBoundary import create_LBM_BC
class LBMGridBoundary(StencilModule):
    '''
    Base Class for defining velocity Boundary conditions for Cell Grids
    '''
    def __init__(self, grid:Grid, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, grid.dimension, grid.dimension, dynamic_array_alloc, float_type)
        self.grid.set_Faces()
        self.faces = self.grid.faces
        self.groups = self.faces.boundary_groups
        
        assert grid.is_LBM, 'LBM values must be invoked before this module is called. use grid.set_LBM()'
        self.latticeModel = grid.LBM_lattice
        self.units = grid.LBM_units
        
        self.boundary_value = np.zeros(shape = (self.faces.num_boundary_faces,self.dimension),dtype=self.float_type)
        self.boundary_type = np.zeros(shape = (self.faces.num_boundary_faces,self.dimension),dtype= np.int8)
        
        self.kernel = create_LBM_BC(grid.dimension,self.latticeModel)
    
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
        
    
    def no_slip(self,face_ids:str|int|np.ndarray|list|tuple):
        self.set_BC(face_ids,0.,0,None)    
    
    
    def moving_wall(self,face_ids:str|int|np.ndarray|list|tuple,value:float,axis):
        value = value/self.units.cU
        axes = tuple([i for i in range(self.dimension)])
        other_axes = axes[:axis] + axis[axis:]
        
        self.set_BC(face_ids,value,1,axis)
        self.set_BC(face_ids,value,1,other_axes)
    
    def velocity_inlet(self):
        pass
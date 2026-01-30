from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector
# from .types import *
from ..Stencil.hooks import *
import numpy as np

class Boundary(ExplicitUniformGridStencil):
    '''
    Base Class for Boundary Methods. For users, they need to add methods to calculate the indices after which they can pass these
    indices to define_boundary_value_and_type_arrays to create the necessary boundary value and type arrays
    '''
    boundary_type: np.ndarray
    boundary_value: np.ndarray
    groups:dict[str,np.ndarray[int]] = dict()
    def __init__(self,field,dx,ghost_cells:int):
        inputs = self.get_shape_from_dtype(field.dtype)
        super().__init__(inputs ,inputs, dx, field.dtype._wp_scalar_type_)
        assert type_is_vector(self.input_dtype), 'input must be vector type'
        assert type(ghost_cells) is int and ghost_cells > 0
        self.ghost_cells = ghost_cells
        self.grid_shape_without_ghost = self.grid_shape_with_no_ghost_cells(field.shape,ghost_cells)
        '''
        Shape of field with no ghost cells
        '''
        self.grid_shape = field.shape
        '''
        Shape of field including ghost cells
        '''
        self.dimension = self.calculate_dimension_from_grid_shape(self.grid_shape)
    
    
    def define_boundary_value_and_type_arrays(self,indices):
        '''
        Create the numpy array attributes boundary_value and boundary_type arrays for the field and set the 'ALL' Group to the indices passed in
        '''
        shape = (len(indices),) + self.input_dtype_shape
        self.boundary_ids = np.arange(len(indices))
        self.boundary_value = np.zeros(shape,dtype=wp.dtype_to_numpy(self.input_scalar_type))
        self.boundary_type = np.zeros_like(self.boundary_value,dtype = np.int8)
        self.groups['ALL'] = self.boundary_ids
        
    
    def _check_output_ids(self,output_ids:int|np.ndarray|list|tuple|None):
        
        if output_ids is None:
            return slice(None)
    
        if isinstance(output_ids,int):
            assert type_is_vector(self.input_dtype)
            assert 0 <= output_ids < self.inputs[0]
            return output_ids
            
        if isinstance(output_ids,(list,tuple,np.ndarray)):
            output_ids = np.array(output_ids,dtype = np.int32)
            assert np.all( 0 <= output_ids < self.inputs)
            return output_ids
        
        raise TypeError(f'Valid Types are: int|np.ndarray|list|tuple|None got {type(output_ids)} instead')
    
    def set_BC(self,face_ids:str|int|np.ndarray|list|tuple,value:float,boundary_type:int,outputs_ids:int|np.ndarray|list|tuple|None):
        '''
        Key:
            0 -> Dirichlet
            1 -> Von Neumann
        '''
        
        if isinstance(face_ids,str):
            assert face_ids in self.groups.keys()
            face_ids = self.groups[face_ids]
            
        assert isinstance(face_ids,(np.ndarray,list,tuple,int))
        assert isinstance(value,float), 'Value must be type float'
        
        outputs_ids = self._check_output_ids(outputs_ids)
        
        self.boundary_type[face_ids,outputs_ids] = boundary_type # For Dirichlet
        self.boundary_value[face_ids,outputs_ids] = value
        
    def dirichlet_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,0,outputs_ids)        
    
    def vonNeumann_BC(self,group:str|int|np.ndarray|list|tuple,value:float,outputs_ids:int|np.ndarray|list|tuple|None = None):
        self.set_BC(group,value,1,outputs_ids)
        
from ..Stencil.stencil import Stencil
import warp as wp
from warp.types import vector,matrix
from ..utils import *
from ..Stencil.hooks import *
from collections.abc import Iterable

class ExplicitUniformGridStencil(Stencil):
    '''
    Class For Stencil on Uniform Grids for AoS grids
    '''
    def __init__(self, inputs:int|list[int],outputs:int|list[int],dx:float,ghost_cells = 0,float_dtype:wp.float32|wp.float64 = wp.float32):
        self.dx = float_dtype(dx)
        self._inputs = tuplify(inputs)
        self._outputs = tuplify(outputs)
        self.ghost_cells = ghost_cells
        input_dtype = self._get_dtype_from_shape(self.inputs,float_dtype)
        output_dtype = self._get_dtype_from_shape(self.outputs,float_dtype)
        super().__init__(input_dtype,output_dtype,float_dtype)
        
    
    
    @property
    def inputs(self) -> tuple[int]:
        assert 1 <= len(self._outputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._inputs
    @property
    def outputs(self) -> tuple[int]:
        assert 1 <= len(self._outputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._outputs

    @property
    def input_dtype_shape(self):
        '''
        Shape of input dtype returned as a tuple
        '''
        return self.get_shape_from_dtype(self.input_dtype)
    
    @property
    def output_dtype_shape(self):
        '''
        Shape of output dtype returned as a tuple
        '''
        return self.get_shape_from_dtype(self.output_dtype)
    
    @property
    def output_scalar_type(self):
        '''
        warp scalar dtype (e.g. float32) of input dtype
        '''
        return self.output_dtype._wp_scalar_type_
    
    @property
    def input_scalar_type(self):
        '''
        warp scalar dtype (e.g. float32) of output dtype
        '''
        return self.output_dtype._wp_scalar_type_
    
    
    @staticmethod
    def calculate_dimension_from_grid_shape(shape):
        return sum([1 for s in shape if s > 1])
    
    
    @staticmethod
    def _get_dtype_from_shape(shape:tuple[int],float_dtype):
        return dtype_from_shape(shape,float_dtype)
        
    @staticmethod
    def get_shape_from_dtype(dtype):
        '''
        Get the shape of the dtype as a tuple based on if dtype is vec or matrix
        '''
        if wp.types.type_is_vector(dtype):
            return (dtype._length_,)
        elif wp.types.type_is_matrix(dtype):
            return dtype._shape_
        
        else:
            raise TypeError('Dtypes supported are warp vector and matrix only')
    

    
    @staticmethod
    def get_ghost_shape_from_stencil(grid_shape,stencil):
        length = stencil._length_
        assert (length % 2) == 1,'stencil must be odd sized'
        num_ghost_cells = (length -1)
        shape = tuple(s-num_ghost_cells for s in grid_shape if s > 1)
        return shape
    
    
    @staticmethod
    def grid_shape_with_no_ghost_cells(grid_shape,ghost_cells):
        '''
        Given ghost_cells, calculate field shape with no ghost cells
        '''
        return tuple(axis - ghost_cells*2 if axis > 1 else axis for axis in (grid_shape))
    
    @staticmethod
    def get_ghost_shape(grid_shape,ghost_cells):
        return tuple(axis + ghost_cells*2 if axis > 1 else axis for axis in (grid_shape))
    
    
    @setup(order = -1)
    def initialize_array(self,input_array,*args,**kwargs):
        self.output_array = self.create_output_array(input_array)
    
    @setup(order = 1)
    def initialize_kernel(self,input_array,*args,**kwargs):
        ...
        
    def forward(self,input_array,*args, **kwargs):
        raise NotImplementedError()
    
        
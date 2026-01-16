from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix
from .utils import *
from .hooks import *
from collections.abc import Iterable

class ExplicitUniformGridStencil(Stencil):
    '''
    Class For Stencil on Uniform Grids for AoS grids
    '''
    def __init__(self, inputs:int|list[int],outputs:int|list[int],dx:float,float_dtype:wp.float32|wp.float64 = wp.float32):
        self.dx = float_dtype(dx)
        self._inputs = tuplify(inputs)
        self._outputs = tuplify(outputs)
        input_dtype = self._get_dtype_from_shape(inputs,float_dtype)
        output_dtype = self._get_dtype_from_shape(outputs,float_dtype)
        super().__init__(input_dtype,output_dtype,float_dtype)
        
    @staticmethod
    def _get_dtype_from_shape(shape:tuple[int],float_dtype):
        
        if isinstance(shape,Iterable):
            assert all([isinstance(x,int) for x in shape]), 'contents in input/output must be int only'
        else:
            assert isinstance(shape,int)
            shape = tuplify(shape)
            
        
        if len(shape) == 1:
            return vector(length = shape[0],dtype = float_dtype)
        else:
            return matrix(shape = shape, dtype = float_dtype)
    
    @property
    def inputs(self) -> tuple[int]:
        assert 1 <= len(self._outputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._inputs
    @property
    def outputs(self) -> tuple[int]:
        assert 1 <= len(self._outputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._outputs
    
    @staticmethod
    def get_ghost_shape(input_shape,stencil):
        length = stencil._length_
        assert (length % 2) == 1,'stencil must be odd sized'
        num_ghost_cells = (length -1)
        shape = tuple(s-num_ghost_cells for s in input_shape if s > 1)
        return shape
    
    @setup(order = -1)
    def initialize_array(self,input_array,*args,**kwargs):
        self.output_array = self.create_output_array(input_array)
    
    @setup(order = 1)
    def initialize_kernel(self,input_array,*args,**kwargs):
        ...
    
    @before_forward
    def zero_array(self,*args,**kwargs):
        self.output_array.zero_()
        
        
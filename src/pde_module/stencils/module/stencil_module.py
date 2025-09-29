from pde_module.grids import NodeGrid
from collections import deque
import warp as wp

import numpy as np


class StencilModule():
    '''
    Base Module Class for Stencil Modules
    '''
    buffer:deque
    levels: int | tuple
    dimension:int
    grid:NodeGrid
    
    requires_grad:bool = False
    dynamic_array_alloc:bool = True
    '''
    If True Always allocate a new array
    '''
    _output_array:wp.array | None = None
    num_inputs:int
    num_outputs:int
    
    levels:int
    output_dtype:wp.vec3f
    input_dtype: wp.vec3f
    

    def __init__(self,grid:NodeGrid,num_inputs:int,num_outputs:int,dynamic_array_alloc:bool = True,float_type = wp.float32):
        self.num_inputs = num_inputs
        self.dimension = grid.dimension
        self.float_type = float_type
        self.grid = grid
        self.dynamic_array_alloc = dynamic_array_alloc
        
        
        self.input_dtype = wp.vec(length=num_inputs,dtype= float_type)
        self.output_dtype = wp.vec(length=num_outputs,dtype= float_type)
        
    
    def output_array(self,input_array:wp.array,dtype = None,zero_array = True):
        '''
        Create the output_array based on the input array
        
        dtype: dtype of output array, if None the dtype is the same as the input array
        zero_array: bool, whether to zero the array before returning, default is True
        '''
                
        if self._output_array is None or self.dynamic_array_alloc:
            self.init_output_array(input_array,dtype)
        
        assert input_array.shape == self._output_array.shape, 'Currently Input array shape cannot change after initialization'
        
        if zero_array:
            self._output_array.zero_()
        
        return self._output_array
        
        
        
    def init_output_array(self,input_array:wp.array,dtype=None):
        if dtype is None:
            dtype = self.output_dtype
        self._output_array = wp.empty(shape= input_array.shape,dtype=dtype) 
        
    
    
    def to_warp(self):
        pass
    def to_numpy(self):
        pass

    def __call__(self, *args, **kwargs):
        output = self.forward(*args,**kwargs)
        return output
        
        
    def forward(self,*args,**kwargs):
        pass
    
    
    
    def __add__(self,other):
        raise NotImplementedError()
    
    def __sub__(self,other):
        raise NotImplementedError()
    def __mul__(self,other):
        raise NotImplementedError()
    
    def __div__(self,other):
        raise NotImplementedError()
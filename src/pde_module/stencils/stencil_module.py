from pde_module.grids import NodeGrid
from pde_module.grids import UniformCellGrid

from pde_module.grids.grid import Grid
from collections import deque
import warp as wp
import warnings
import numpy as np
import inspect



class StencilModule():
    '''
    Base Module Class for Stencil Modules
    '''
    buffer:deque
    levels: int | tuple
    dimension:int
    grid:Grid
    
    requires_grad:bool = False
    dynamic_array_alloc:bool = True
    '''
    If True Always allocate a new array
    '''
    _output_array:wp.array | None = None
    num_inputs:int |tuple| None
    num_outputs:int  |tuple| None
    
    init_stencil_flag:bool = True
    
    levels:int
    output_dtype:wp.vec3f | None = None
    input_dtype: wp.vec3f | None = None
    
    zero_output_array: bool = True


    def __init__(self,grid:Grid,num_inputs:int,num_outputs:int,dynamic_array_alloc:bool = True,float_type = wp.float32):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dimension = grid.dimension
        self.float_type = float_type
        self.grid = grid
        self.dynamic_array_alloc = dynamic_array_alloc
        
        if num_inputs is not None:
            self.set_input_dtype(num_inputs)
        if num_outputs is not None:
           self.set_output_dtype(num_outputs)
    
    def set_output_dtype(self,num_outputs):
        self.num_outputs = num_outputs
        if isinstance(num_outputs,int):
            self.output_dtype = wp.vec(length=num_outputs,dtype= self.float_type)
        elif isinstance(num_outputs,tuple):
            assert len(num_outputs) == 2, 'output shape must be equal to 2'
            self.output_dtype = wp.mat(shape=num_outputs,dtype= self.float_type)
        else:
            raise TypeError('num outputs must be either int for vector or tuple for matrix dtype')
        
    def set_input_dtype(self,num_inputs):
        self.num_inputs = num_inputs
        
        if isinstance(num_inputs,int):
            self.output_dtype = wp.vec(length=num_inputs,dtype= self.float_type)
        elif isinstance(num_inputs,tuple):
            assert len(num_inputs) == 2, 'output shape must be equal to 2'
            self.output_dtype = wp.mat(shape=num_inputs,dtype= self.float_type)
        else:
            raise TypeError('num_inputs must be either int for vector or tuple for matrix dtype')
        
    
    
    @property
    def output_array(self):
        '''
        Return output_array.
        
        by default the array is zeroed each time output_array is called. to disable this set the attribute flag `zero_output_array` to `False` (default is True)
        
        '''
        if self.zero_output_array:
            self._output_array.zero_()
        
        return self._output_array
        
    @property
    def output_type(self):
        '''
        String representing output type. Valid option types are strings: `vector` or `matrix` .If invalid Output type, `None` is returned
        '''
        if isinstance(self.num_outputs,int):
            return 'vector'
        elif isinstance(self.num_outputs,tuple):
            return 'matrix'
        else:
            return None    
    
    @property
    def input_type(self):
        '''
        String representing input field type. Valid option types are strings: `vector` or `matrix` .If invalid input type, `None` is returned
        '''
        if isinstance(self.num_inputs,int):
            return 'vector'
        elif isinstance(self.num_inputs,tuple):
            return 'matrix'
        else:
            return None    
    
    
    
    def init_output_array(self,input_array:wp.array,dtype=None):
        if dtype is None:
            dtype = self.output_dtype
        self._output_array = wp.empty(shape= input_array.shape,dtype=dtype) 
        
    
    def init_stencil(self,*args,**kwargs):
        '''
        Method to call to initialize before forward method is made. This should be explcitely called before creating a CUDA Graph.
        
        init_stencil should have the same input arguements as the foward method or contain the *args and **kwargs arguments
        
        Example:
            - For most stencils, this can be used to initialize the output array given the initial conditions array.
            - For element wise mapping, this method initializes both the output arrays AND the kernel
            - For GridBoundary, to_warp() is called and output array is initialized
        
        '''
        raise NotImplementedError('init_stencil method in Stencil Object must be implemented by the user')
    
    
    def to_warp(self):
        '''
        Method to overide if needed. Default, this method does nothing but raise a warning
        
        Convert any numpy arrays into warp arrays to be used in any associated stencil kernels. Used by say GridBoundary to track what boundary types and values.
        '''
        warnings.warn(f'{inspect.currentframe().f_code.co_name}() method was called but not overidden. This method by default does nothing ')
    def to_numpy(self):
        '''
        Method to overide if needed. Default, this method does nothing but raise a warning
        
        Convert any warp arrays associated with the stencil back into numpy arrays
        '''
        warnings.warn(f'{inspect.currentframe().f_code.co_name}() method was called but not overidden. This method by default does nothing ')

    def __call__(self, *args, **kwargs):
        
        if self.init_stencil_flag:
            self.init_stencil(*args,**kwargs)
            self.init_stencil_flag = False
        
        output = self.forward(*args,**kwargs)
        return output
        
        
    def forward(self,*args,**kwargs):
        '''
        Method to overide. This method should contain all the code neccesary to caclulate the stencil output. Default, this method raises an exception
        '''
        raise NotImplementedError('forward method in Stencil Object must be implemented by the user')
    
    
    
    def __add__(self,other):
        raise NotImplementedError()
    
    def __sub__(self,other):
        raise NotImplementedError()
    def __mul__(self,other):
        raise NotImplementedError()
    
    def __div__(self,other):
        raise NotImplementedError()
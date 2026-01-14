import warp as wp
from warp import types
from .types import *

from collections.abc import Iterable
def tuplify(x:Any):
    '''Convert single item into tuple of element one or tuple. String and bytes are treated as a single object. All other iterables converted to tuple'''
    if isinstance(x,(str,bytearray,bytes)):
        return (x,)
    elif isinstance(x,Iterable):
        return tuple(x)
    else:
        return (x,)

class Stencil:
    '''
    Array of Structure Stencil Base Module.
    
    Think of it like nn.Module from Pytorch does the following:
    - Identify the appropriate dtype for input and output
    - Initialise and store the output array
    
    TODO:
    - Register stencil like nn.Module (Far into future)
    - Add grad for autodiff
    '''
    step:int = 0
    float_dtype: wp.float32 | wp.float64 
    def __init__(self,inputs:int|list[int]|tuple[int],outputs:int|list[int]|tuple[int],float_dtype:float = wp.float32):
        self._inputs = tuplify(inputs)
        self._outputs = tuplify(outputs)
        self.float_dtype = float_dtype
        self.initial = True
        self.zero_array = True
        
    @property
    def inputs(self) -> tuple[int]:
        assert 1 <= len(self._inputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._inputs
    @property
    def outputs(self) -> tuple[int]:
        assert 1 <= len(self._outputs) <= 2, 'AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass'  
        return self._outputs
    
    @property
    def output_dtype(self) -> wp_Matrix | wp_Vector:
        priv_attr = '_output_dtype'
        if hasattr(self,priv_attr) is False:
            
            if len(self.outputs) == 1:
                value = wp.types.vector(length = self.inputs[0],dtype = self.float_dtype)    
            else:
                value = wp.types.matrix(shape = self.inputs,dtype = self.float_dtype)
                
            setattr(self,priv_attr,value)
        return getattr(self,priv_attr)
    @property
    def input_dtype(self) ->wp_Matrix | wp_Vector :
        priv_attr = '_input_dtype'
        if hasattr(self,priv_attr) is False:
            
            if len(self.outputs) == 1:
                value = wp.types.vector(length = self.inputs[0],dtype = self.float_dtype)    
            else:
                value = wp.types.matrix(shape = self.inputs,dtype = self.float_dtype)
                
            setattr(self,priv_attr,value)
        return getattr(self,priv_attr)
    
    def create_output_array(self,input_array:wp.array):
        shape = input_array.shape
        return wp.empty(shape = shape, dtype= self.output_dtype)
    
    
    def forward(self,input_array,*args,**kwargs):
        '''
        Method to overide. This method should contain all the code neccesary to caclulate the stencil output. Default, this method raises an exception
        '''
        raise NotImplementedError('forward method in Stencil Object must be implemented by the user')
    
    
    def setup(self,input_array,*args,**kwargs):
        '''
        Call this function to safely call initialize outside of call function to gurantee setting the initial flag
        '''
        assert wp.types.types_equal(input_array.dtype,self.input_dtype), 'Input array dtype must match stencil given dtype'
        self.initialize_array(input_array,*args,**kwargs)
        self.initialize_kernel(input_array,*args,**kwargs)
        self.initial = False
        
    def initialize_kernel(input_array,*args,**kwargs):
        '''
        Method to overide. This method should contain the setting needed initialize. Example use is then the kernel is dependent on grid shape e.g. ghost cells. Default does nothing
        '''
        ...
    
    def initialize_array(self,input_array,*args,**kwargs):
        '''
        Hook to perform on first pass. By default it creates the output array and caches it so it is not reloaded each time. Is called only once
        '''
        self.output_array = self.create_output_array(input_array)
    
    
    def before_forward(self,input_array,*args,**kwargs):
        '''
        Hook to do operation before forward pass is called. By Default this will zero the output array. If this is undesirable, overide this method
        '''
        self.output_array.zero_()
    
    def after_forward(self,input_array,*args,**kwargs):
        '''
        Hook to do operation after forward pass is called. By default, does nothing
        '''
        ...
    
    def __call__(self,input_array,*args,**kwargs):
        '''
        Call Function to Stencil. Does the following:
        1. Setup (if step == 0)(default: create output array and create kernel.)
        2. Before Forward Call (default: zero output array)
        3. Forward call - Do your calcs here (MUST BE IMPLEMENTED)
        4. After Forward Call (default does nothing)
        5. increment step counter by 1
        '''
        if self.initial:
            self.setup(input_array,*args,**kwargs)
            self.initial = False
        self.before_forward(input_array,*args,**kwargs)
        output = self.forward(input_array,*args,**kwargs)
        self.after_forward(input_array,*args,**kwargs)
        self.step += 1
        return output
        
    
if __name__ == '__main__':
    x = wp.array()
    wp.types.vector()
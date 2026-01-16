import warp as wp
from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix
from .hooks import *
'''
Modules for Time Integration Schemes. These Stencils should be grid and float type agnostic 
'''

class ForwardEuler(Stencil):
    def __init__(self, input_dtype, float_dtype = wp.float32):
        super().__init__(input_dtype,input_dtype, float_dtype)
    
    @setup
    def initialize_kernel(self,input_array,*args,**kwargs):
        self.output_array = super().create_output_array(input_array)
        ndim = len(input_array.shape)
        self.kernel = create_forward_euler(self.input_dtype,ndim,self.float_dtype)    
    
    
    def forward(self,input_array,stencil_values,dt):
        assert input_array.shape == stencil_values.shape == self.output_array.shape
        wp.launch(kernel=self.kernel,dim = input_array.shape,inputs = [input_array,stencil_values,dt], outputs = [self.output_array])


def create_forward_euler(array_dtype,ndim,float_dtype):
    assert ndim <= 4 ,'Max numer of ndim for warp arrays is 4'
    array_type = wp.array(ndim = ndim,dtype = array_dtype)
    
    @wp.kernel
    def forward_euler_kernel(
                        current_values:array_type,
                        stencil_values:array_type,
                        dt:float_dtype,
                        new_values:array_type,
    ):
        if wp.static(ndim == 1):
            tid = wp.tid() 
            new_values[tid] = current_values[tid] + dt*stencil_values[tid]
        elif wp.static(ndim == 2):
            x,y = wp.tid()
            new_values[x,y] = current_values[x,y] + dt*stencil_values[x,y]
        elif wp.static(ndim == 3):
            x,y,z = wp.tid()
            new_values[x,y,z] = current_values[x,y,z] + dt*stencil_values[x,y,z]
        else: 
            x,y,z,w = wp.tid()
            new_values[x,y,z,w] = current_values[x,y,z,w] + dt*stencil_values[x,y,z,w]
        
    return forward_euler_kernel
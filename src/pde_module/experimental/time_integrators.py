import warp as wp
from .Stencil.stencil import Stencil
import warp as wp
from warp.types import vector,matrix
from .Stencil.hooks import *
'''
Modules for Time Integration Schemes. These Stencils should be grid and float type agnostic 
'''

class ForwardEuler(Stencil):
    def __init__(self, input_dtype):
        self.input_dtype = input_dtype
        self.float_dtype = input_dtype._wp_scalar_type_
        super().__init__()
    
    
    @setup
    def initialize_kernel(self,input_array,*args,**kwargs):
        self.output_array = super().create_output_array(input_array)
        self.kernel = create_forward_euler(self.input_dtype,self.float_dtype)    
        self.size = input_array.size
    
    def forward(self,input_array,stencil_values,dt):
        assert input_array.shape == stencil_values.shape == self.output_array.shape
        wp.launch(kernel=self.kernel,dim = self.size,inputs = [input_array.flatten(),stencil_values.flatten(),dt], outputs = [self.output_array.flatten()])
        return self.output_array

def create_forward_euler(array_dtype,float_dtype):
    array_type = wp.array(dtype = array_dtype)
    @wp.kernel
    def forward_euler_kernel(
                        current_values:array_type,
                        stencil_values:array_type,
                        dt:float_dtype,
                        new_values:array_type,
    ):
        
        tid = wp.tid() 
        new_values[tid] = current_values[tid] + dt*stencil_values[tid]
    
    return forward_euler_kernel
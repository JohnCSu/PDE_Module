import warp as wp
from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix

'''
Modules for Time Integration Schemes. These Stencils should be grid and float type agnostic 
'''

class ForwardEuler(Stencil):
    def __init__(self, inputs, outputs, float_dtype = wp.float32):
        super().__init__(inputs, outputs, float_dtype)
        
    def initialize_kernel(self,input_array,*args,**kwargs):
        dims = len(input_array.shape)
        self.kernel = create_forward_euler(self.input_dtype,dims,self.float_dtype)    
    
    def forward(self,input_array,stencil_values,dt):
        wp.launch(self.kernel,input_array.shape,inputs = [input_array,stencil_values,dt], outputs = [self.output_array])


def create_forward_euler(array_dtype,dims,float_dtype = wp.float32):
    assert dims <= 4 ,'Max numer of dims for warp arrays is 4'
    @wp.kernel
    def forward_euler_kernel(
                        current_values:wp.array(dims = dims,dtype = array_dtype),
                        stencil_values:wp.array(dims = dims,dtype = array_dtype),
                        dt:float_dtype,
                        new_values:wp.array(dims = dims,dtype = array_dtype),
    ):
        tid = wp.tid() # Lets only do internal grid points
        new_values[tid] = current_values[tid] + dt*stencil_values[tid]
    
    return forward_euler_kernel
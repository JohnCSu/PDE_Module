from ...stencil_module import StencilModule
import warp as wp
import numpy as np
from pde_module.utils.type_check import is_dtype_wp_vector

class ElementWiseMap(StencilModule):
    '''
    Dynamically create and store a kernel that is created from wp.map. Ideal for element wise operations (think like relu) where output.shape == input.shape
    
    Assumptions:
        - The first argument input to the function to be mapped must be an array, subsequent input arguments can be arbitary
        - Fist array must have a vector dtype, subsequent arguments can be any type
        - Output array shape must match input array shape and be of vector dyype (length of vector dtype can differ from input array)
        - If the output array is not specified, then it is assumed that the output array matches the input array
        - Once the kernel is created, it can be reset by setting the create_kernel flag to True again
    
    Useful as this will store the kernel created from the mapping preventing constant reloading and maintain the output array automoatically, without
    dynamically reallocating a new one each time, also using lambda function can easily fuse operations together for element wise operations
    
    Note that due to how wp.map works, a new kernel is created everytime 
     

    '''
    def __init__(self,func:str,grid,num_outputs = None, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, None, num_outputs, dynamic_array_alloc, float_type)
        self.func = func
        
    
    def init_stencil(self, *args):
        self.init_stencil_flag = False
        input_array = args[0]
        assert isinstance(input_array,wp.array), 'First argument input must be a warp array'
        assert is_dtype_wp_vector(input_array), 'First input array dtype must be a vector (scalar valus should be vector of size 1)'
    
        #set input array shape
        self.set_input_dtype(input_array.dtype._length_)
        
        # Set Output array shapes
        if self.num_outputs is None:
            self.num_outputs = self.num_inputs
        
        self.set_output_dtype(self.num_outputs)
        self.init_output_array(input_array)
        # else:
        #     assert is_dtype_wp_vector(output_array), 'Output array dtype must be a vector (scalar values should be vector of size 1)'
        #     self.set_output_dtype(len(output_array.dtype))
        #     self._output_array = output_array  
        self.kernel = wp.map(self.func,*args,out = self._output_array,return_kernel= True)
        
    def forward(self,*args):
        '''
        on first call, create and store the kernel and on subsequent calls only launch said kernel.
        
        The first input argument must be an array with wp.vec dtype
        
        ''' 
        
        input_array = args[0]
        output_array = self.output_array
        threads_shape = (input_array.shape[0],) + (self.grid.shape)
        wp.launch(self.kernel,dim = threads_shape,inputs= args,outputs= [output_array])
        return output_array
        
        
    
    
from .Stencil.stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix
from .Stencil.hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift


class ElementWise(Stencil):
    '''
    Base_class for Element wise operations between two arrays of same size
    '''
    def __init__(self,element_op,output_dtype,float_dtype = wp.float32):
        self.element_op = element_op
        self._output_dtype
        self.float_dtype = float_dtype
        
        
    @setup(order = 1)
    def initialize_kernel(self,array_A,array_B,*args, **kwargs):
        assert array_A.shape == array_B.shape, 'input arrays must be the same!'
        super().__init__(array_A.dtype,self._output_dtype,0,0,self.float_dtype)
        self.array_B_dtype =array_B.dtype
        
        self.kernel = create_ElementOp_kernel(self.element_op,array_A.dtype,array_B.dtype,self.output_dtype)
        self.create_output_array(array_A)
    
    
    def forward(self, array_A:wp.array,array_B:wp.array,*args,**kwargs):    
        dim = array_A.flatten().shape
        wp.launch(self.kernel,dim = dim,inputs = [
            array_A.flatten(),
            array_B.flatten(),   
        ],
        outputs= [self.output_array])
        return self.output_array


def create_ElementOp_kernel(element_op,array_A_dtype,array_B_dtype,output_array_dtype):
    
    @wp.kernel
    def elementWise_kernel(A:wp.array(dtype=array_A_dtype),
                           B:wp.array(dtype=array_B_dtype),
                           output_array:wp.array(dtype = output_array_dtype)):
        i = wp.tid()
        output_array[i] =  element_op(A[i],B[i])
        

    
    return elementWise_kernel
        
        
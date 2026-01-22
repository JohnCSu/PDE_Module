from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix
from .hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift



class ElementWise(Stencil):
    '''
    Base_class for Element wise operations
    '''
    def __init__(self,element_op,output_dtype):
        self.element_op = element_op
        self._output_dtype
        # super().__init__(dimension, 1,dx,ghost_cells,float_dtype=float_dtype)
        
    @setup
    def initialize_kernel(self,array_A,array_B,*args, **kwargs):
        
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        
        self.kernel = create_ElementOp_kernel(self.element_op,array_A.dtype,array_B.dtype,)
        
    
    def forward(self, array_A,array_B,*args,**kwargs):    
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
        
        
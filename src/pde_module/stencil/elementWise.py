from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from .hooks import *




class ElementWise(Stencil):
    '''
    Base_class for Element wise operations between two arrays of same size
    '''
    def __init__(self,element_op,output_dtype = None):
        self.element_op = element_op
        self.output_dtype = output_dtype
        super().__init__()
            
    @setup(order = 1)
    def initialize_kernel(self,array_A,array_B,*args, **kwargs):
        assert array_A.shape == array_B.shape, 'input arrays must be the same Shape!'
        self.array_B_dtype =array_B.dtype
        self.array_A_dtype =array_A.dtype
        
        if self.output_dtype is None:
            self.output_dtype = self.array_A_dtype
            
        self.kernel = create_ElementOp_kernel(self.element_op,array_A.dtype,array_B.dtype,self.output_dtype)
        self.output_array = self.create_output_array(array_A,self.output_dtype)
        
    def forward(self, array_A:wp.array,array_B:wp.array,*args,**kwargs):    
        dim = array_A.size
        wp.launch(self.kernel,dim = dim,inputs = [
               array_A.flatten(),array_B.flatten()
        ],
        outputs= [self.output_array.flatten()])
        return self.output_array


def create_ElementOp_kernel(element_op,array_A_dtype,array_B_dtype,output_array_dtype):
    
    @wp.kernel
    def elementWise_kernel(A:wp.array(dtype=array_A_dtype),
                           B:wp.array(dtype=array_B_dtype),
                           output_array:wp.array(dtype = output_array_dtype)):
        i = wp.tid()
        output_array[i] = element_op(A[i],B[i])
        
        
    return elementWise_kernel
        
        
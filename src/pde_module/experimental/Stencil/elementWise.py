from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from .hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift




class ElementWise(Stencil):
    '''
    Base_class for Element wise operations between two arrays of same size
    '''
    def __init__(self,element_op,output_dtype,float_dtype = wp.float32,debug = False):
        self.element_op = element_op
        self.float_dtype = float_dtype
        super().__init__(output_dtype,output_dtype,self.float_dtype,debug = False)
            
    @setup(order = 1)
    def initialize_kernel(self,array_A,array_B,*args, **kwargs):
        assert array_A.shape == array_B.shape, 'input arrays must be the same Shape!'
        self.array_B_dtype =array_B.dtype
        self.array_A_dtype =array_A.dtype
        
        self.kernel = create_ElementOp_kernel(self.element_op,array_A.dtype,array_B.dtype,self.output_dtype)
        self.output_array = self.create_output_array(array_B)
        
    def forward(self, array_A:wp.array,array_B:wp.array,*args,**kwargs):    
        dim = array_A.size
        x,y = array_A.flatten(),array_B.flatten()
        z = self.output_array.flatten()
        wp.launch(self.kernel,dim = dim,inputs = [
               x,y
        ],
        outputs= [z])
        return self.output_array


def create_ElementOp_kernel(element_op,array_A_dtype,array_B_dtype,output_array_dtype):
    
    @wp.kernel
    def elementWise_kernel(A:wp.array(dtype=array_A_dtype),
                           B:wp.array(dtype=array_B_dtype),
                           output_array:wp.array(dtype = output_array_dtype)):
        i = wp.tid()
        output_array[i] = element_op(A[i],B[i])
        
        
    return elementWise_kernel
        
        
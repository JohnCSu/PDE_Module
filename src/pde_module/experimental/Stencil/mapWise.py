from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from .hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift


class MapWise(Stencil):
    '''
    Base_class for Map_wise operations. The first input must be an array and the output array
    is assumed to be the same dtype and shape
    '''
    def __init__(self,element_op):
        self.element_op = element_op
        super().__init__()
        
    @setup(order = 1)
    def initialize_kernel(self,array_A,*args, **kwargs):
        self.input_dtype = array_A.dtype
        self.output_dtype = array_A.dtype
        self.output_array = self.create_output_array(array_A)
        self.kernel = wp.map(self.element_op,array_A,*args,out= self.output_array,return_kernel=True)
        
    def forward(self, array_A:wp.array,*args,**kwargs):    
        dim = array_A.shape
        wp.launch(self.kernel,dim = dim,inputs = [
        array_A,*args       
        ],
        outputs= [self.output_array])
        return self.output_array

        
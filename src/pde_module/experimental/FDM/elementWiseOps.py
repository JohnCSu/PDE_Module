from ..Stencil.elementWise import ElementWise

import warp as wp
from warp.types import vector,matrix,types_equal
from ..Stencil.hooks import *
from ..utils import dtype_from_shape


class OuterProduct(ElementWise):
    '''
    Compute outer product between 2 vectors and output as matrix
    '''
    def __init__(self, vec_A:int,vec_B:int, float_dtype=wp.float32):
        element_op = wp.outer
        output_dtype = matrix((vec_A,vec_B),dtype = float_dtype)
        super().__init__(element_op, output_dtype, float_dtype)
        self.vec_A = vector(vec_A,float_dtype)
        self.vec_B = vector(vec_B,float_dtype)
    @setup(order = 2)
    def check_same(self,array_A,array_B,*args, **kwargs):
        assert types_equal(array_A.dtype,self.vec_A) and types_equal(array_B.dtype,self.vec_A)
        



class scalarVectorMult(ElementWise):
    '''
    Multiply a scalar field (i.e. vector with length 1) with a corresponding vector/matrix field (which can be arbitary).
    '''
    def __init__(self, outputs, float_dtype=wp.float32):
        output_dtype = dtype_from_shape(outputs,float_dtype)
        element_op = scalarVectorMultiply(output_dtype)
        super().__init__(element_op, output_dtype, float_dtype)


def scalarVectorMultiply(output_dtype):
    float_type = output_dtype._wp_scalar_type_
    @wp.func
    def _scalarVectorMultiply(a:vector(1,float_type),b:output_dtype):
        return a[0]*b
        
    return _scalarVectorMultiply


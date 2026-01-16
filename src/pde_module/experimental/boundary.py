from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix
# from .types import *
from .hooks import *


class GridBoundary(ExplicitUniformGridStencil):
    def __init__(self, inputs,num_ghost:int,dx, float_dtype = wp.float32):
        super().__init__(inputs, inputs, dx, float_dtype)
        self.num_ghost = num_ghost
    
        
    def get_boundary_indices(self,field):
        self.ghost_shape = self.get_ghost_shape(field.shape,self.num_ghost)
        
        
        
def create_boundary_kernel(float_dtype):
    
    @wp.kernel
    def boundary(
        current_values:wp.array3d(dtype = input_dtype),
        dx:float_dtype,
        new_values:wp.array3d(dtype = input_dtype),
        ):
        
        pass
        
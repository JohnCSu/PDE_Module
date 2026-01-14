from .stencil import Stencil
import warp as wp
from warp.types import vector,matrix

class UniformGridStencil(Stencil):
    '''
    Stencil Class with useful function for ops on uniform grids
    '''
    def __init__(self, inputs:int|list[int],outputs:int|list[int],dx:float, float_dtype = wp.float32):
        super().__init__(inputs, outputs, float_dtype)
        self.dx = self.float_dtype(dx)
    @staticmethod
    def get_ghost_shape(input_shape,stencil):
        length = stencil._length_
        assert (length % 2) == 1,'stencil must be odd sized'
        num_ghost_cells = (length -1)
        shape = tuple(s-num_ghost_cells for s in input_shape if s > 1)
        return shape
        
        
        
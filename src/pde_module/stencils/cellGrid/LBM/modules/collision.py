from ...stencil_module import StencilModule
import warp as wp

class Collision(StencilModule):
    
    def __init__(self, grid, num_inputs, num_outputs, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, num_inputs, num_outputs, dynamic_array_alloc, float_type)
    
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def init_stencil(self, *args, **kwargs):
        return super().init_stencil(*args, **kwargs) 
    
    
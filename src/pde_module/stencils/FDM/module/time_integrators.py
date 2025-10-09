from ..functional.time_integrators import forward_euler
from ..kernels.time_integrators import create_forward_euler_kernel

from ...stencil_module import StencilModule
from pde_module.grids.node_grid import NodeGrid


class ForwardEuler(StencilModule):
    def __init__(self, grid, num_inputs, dynamic_array_alloc = True, **kwargs):
        super().__init__(grid, num_inputs,num_inputs, dynamic_array_alloc, **kwargs)

        self.kernel = create_forward_euler_kernel(num_inputs)
        
    def forward(self,input_array,stencil_values,dt):
        return forward_euler(self.kernel,input_array,self.output_array,stencil_values,dt)
    
    
    def init_stencil(self,input_array,stencil_values,dt):
        self.init_output_array(input_array)
        self.init_stencil_flag = False
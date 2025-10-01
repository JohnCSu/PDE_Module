import warp as wp
from ..kernels.derivatives import create_second_derivative_kernel
from ..functional.derivatives import second_derivative
from ...stencil_module import StencilModule
from pde_module.grids.node_grid import NodeGrid

'''
Derivative Kernels defined up to second order.
'''

class Second_Derivative(StencilModule):
    def __init__(self, grid:NodeGrid,num_outputs,axis,levels,buffer_size = 0,dynamic_array_alloc = True,**kwargs):
        super().__init__(grid,num_outputs, levels,buffer_size,dynamic_array_alloc,**kwargs)
        self.kernel = create_second_derivative_kernel(axis,self.num_outputs,levels)
        
    def forward(self,current_values,alpha):
        return second_derivative(self.kernel,self.grid,current_values,self.new_values(current_values),alpha)
    
    
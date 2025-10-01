import warp as wp
from ..kernels.operators import create_Laplacian_kernel,create_grad_kernel,create_Divergence_kernel
from ..functional.operators import laplacian,grad,divergence
from ...stencil_module import StencilModule
from pde_module.grids.node_grid import NodeGrid
class Laplacian(StencilModule):
    '''
    Calculate Laplacian using central difference for each element in the vector field.
    
    
    
    '''
    def __init__(self, grid:NodeGrid,num_inputs,dynamic_array_alloc = True,**kwargs):
        super().__init__(grid,num_inputs,num_inputs,dynamic_array_alloc,**kwargs)
        self.kernel = create_Laplacian_kernel(self.dimension,num_inputs,self.grid.levels)
        
    def forward(self,current_values,scale=1.):
        threads_shape = (current_values.shape[0],) + (self.grid.shape)
        return laplacian(self.kernel,self.grid.stencil_points,threads_shape,current_values,scale,self.grid.levels,self.grid.dimension,self.output_array(current_values))
    


class Curl(StencilModule):
    pass



class Convection(StencilModule):
    '''
    Calculate using first or second order upwind for vector field of size 3
    '''
    pass


class Divergence(StencilModule):
    '''
    Divergence of a Field
    
    inputs:
    grid: NodeGrid object
    num_inputs: int | None, number of inputs. if None, then num_inputs is assumed to mathc grid dimension.
    type: str default: 'vector', str representing the divergence formulation.
        - 'vector' - We apply divergence in a vector field fashion. The number of inputs must match dimension of grid
        - 'tensor' - We apply divergence is a row wise fashion
    
    
    '''
    def __init__(self, grid,num_inputs= None,type = 'vector',*, dynamic_array_alloc = True, **kwargs):
        if num_inputs is None:
            num_inputs = grid.dimension
        
        
        super().__init__(grid,num_inputs,1, dynamic_array_alloc, **kwargs)
            
        self.type = type
        self.kernel = create_Divergence_kernel(grid.dimension,grid.levels,num_inputs)
        
        
        
        
    def forward(self, input_value,scale = 1.):
        threads_shape = (input_value.shape[0],) + (self.grid.shape)
        return divergence(self.kernel,self.grid.stencil_points,threads_shape,input_value,scale,self.grid.levels,self.grid.dimension,self.output_array(input_value))
    
    
class Jacobian(StencilModule):
    '''
    Calculate the jacobian using central difference, take in output vector N at each point and outputs Nx3 matrices.
    if the vector is of size 1 and you want a vector output instead see Grad stencil
    '''
    pass


class Grad(StencilModule):
    '''
    Calculate the grad of a scalar field
    
    inputs:
    grid: NodeGrid object
    num_inputs: int | None, number of inputs. if None, then num_inputs is assumed to mathc grid dimension.
    type: str default: 'vector', str representing the divergence formulation.
        - 'vector' - We apply divergence in a vector field fashion. The number of inputs must match dimension of grid
        - 'tensor' - We apply divergence is a row wise fashion
    
    
    '''
    def __init__(self, grid,num_outputs = None,type = 'vector',*, dynamic_array_alloc = True, **kwargs):
        if num_outputs is None:
            num_outputs = grid.dimension
        super().__init__(grid, 1,num_outputs, dynamic_array_alloc, **kwargs)
        
        self.kernel = create_grad_kernel(grid.dimension,grid.levels,num_outputs)
        
    def forward(self, input_value,scale = 1.):
        threads_shape = (input_value.shape[0],) + (self.grid.shape)
        return grad(self.kernel,self.grid.stencil_points,threads_shape,input_value,scale,self.grid.levels,self.grid.dimension,self.output_array(input_value))
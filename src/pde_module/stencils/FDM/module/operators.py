import warp as wp
from ..kernels.operators import create_Laplacian_kernel,create_grad_kernel,create_Divergence_kernel,create_outer_product_kernel
from ..functional.operators import laplacian,grad,divergence,outer_product
from ...stencil_module import StencilModule
from pde_module.grids.node_grid import NodeGrid
class Laplacian(StencilModule):
    '''
    Calculate Laplacian using central difference for each element in the vector field.
    '''
    def __init__(self, grid:NodeGrid,num_inputs,dynamic_array_alloc = True,**kwargs):
        super().__init__(grid,num_inputs,num_inputs,dynamic_array_alloc,**kwargs)
        self.kernel = create_Laplacian_kernel(self.dimension,num_inputs,self.grid.levels)
        
    def forward(self,input_array,scale=1.):
        threads_shape = (input_array.shape[0],) + (self.grid.shape)
        return laplacian(self.kernel,self.grid.stencil_points,threads_shape,input_array,scale,self.grid.levels,self.grid.dimension,self.output_array)
    
    def init_stencil(self,input_array, *args, **kwargs):
        self.init_stencil_flag = False
        self.init_output_array(input_array)
        

class Curl(StencilModule):
    pass



class Convection(StencilModule):
    '''
    Calculate using first or second order upwind for vector field of size 3
    '''
    pass


class OuterProduct(StencilModule):
    '''
    given 2 Vector fields v1,v2 calculate the outer product to return a matrix field of size (len(v1),len(v2)) 
    
    '''
    def __init__(self, grid, vector_A_Length, vector_B_Length, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, vector_A_Length, (vector_A_Length,vector_B_Length), dynamic_array_alloc, float_type)
        self.kernel = create_outer_product_kernel(vector_A_Length,vector_B_Length)
        
        self.vector_lengths = (vector_A_Length,vector_B_Length)
        
    def forward(self,vector_field_1,vector_field_2,scale = 1.):
        threads_shape = (vector_field_1.shape[0],) + self.grid.shape
        return outer_product(self.kernel,threads_shape,vector_field_1,vector_field_2,scale,self.output_array)

class Convection(StencilModule):
    '''
    Solve the following 
    
    divergence(outer(F1,u_F)) 
    
    where F1 is some arbitary field and u_f is an existing vector field os size grid dimension. outer(*) is the vector outer product
    
    '''

    def __init__(self, grid, num_inputs, dynamic_array_alloc = True, float_type=wp.float32):
        super().__init__(grid, num_inputs, num_inputs, dynamic_array_alloc, float_type)
    
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
        




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
        
        
    def forward(self, input_array,scale = 1.):
        threads_shape = (input_array.shape[0],) + (self.grid.shape)
        return divergence(self.kernel,self.grid.stencil_points,threads_shape,input_array,scale,self.grid.levels,self.grid.dimension,self.output_array)
    
    def init_stencil(self,input_array, *args, **kwargs):
        self.init_stencil_flag = False
        self.init_output_array(input_array)
        
        
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
        
    def forward(self, input_array,scale = 1.):
        threads_shape = (input_array.shape[0],) + (self.grid.shape)
        return grad(self.kernel,self.grid.stencil_points,threads_shape,input_array,scale,self.grid.levels,self.grid.dimension,self.output_array)
    
    
    def init_stencil(self,input_array, *args, **kwargs):
        self.init_stencil_flag = False
        self.init_output_array(input_array)
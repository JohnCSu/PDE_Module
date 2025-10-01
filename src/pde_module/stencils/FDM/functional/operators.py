import warp as wp
from .derivatives import second_derivative
from pde_module.grids import NodeGrid

def laplacian(kernel,stencil_points:wp.array,threads_shape:tuple[int],current_values,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    '''
    Functional Laplacian, the kernel can be found in stencils.kernels:
    
    ```
    from pde_module.stencils.kernels.operators import create_Laplacian_kernel
    kernel = create_Laplacian_kernel(dimension,num_outputs,levels)
    ```
    '''
    wp.launch(kernel,dim = threads_shape,inputs = [stencil_points,current_values,alpha,dimension,levels],outputs = [new_values])
    return new_values
    
def grad(kernel,stencil_points:wp.array,threads_shape:tuple[int],current_values,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    wp.launch(kernel,dim = threads_shape,inputs = [stencil_points,current_values,alpha,dimension,levels],outputs = [new_values])
    return new_values
    
def divergence(kernel,stencil_points:wp.array,threads_shape:tuple[int],current_values,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    '''Function to calculate the diveregence of a 3D Vector field'''
    wp.launch(kernel,dim = threads_shape,inputs = [stencil_points,current_values,alpha,dimension,levels],outputs = [new_values])
    return new_values
    

    
    
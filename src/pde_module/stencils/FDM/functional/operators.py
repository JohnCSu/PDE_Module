import warp as wp
from .derivatives import second_derivative
from pde_module.grids import NodeGrid

def laplacian(kernel,threads_shape:tuple[int],current_values,dx:float,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    '''
    Functional Laplacian, the kernel can be found in stencils.kernels:
    
    ```
    from pde_module.stencils.kernels.operators import create_Laplacian_kernel
    kernel = create_Laplacian_kernel(dimension,num_outputs,levels)
    ```
    '''
    wp.launch(kernel,dim = threads_shape,inputs = [current_values,dx,alpha,dimension,levels],outputs = [new_values])
    return new_values
    
def grad(kernel,threads_shape:tuple[int],current_values,dx:float,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    wp.launch(kernel,dim = threads_shape,inputs = [current_values,dx,alpha,dimension,levels],outputs = [new_values])
    return new_values
    
def divergence(kernel,threads_shape:tuple[int],current_values,dx:float,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    '''Function to calculate the diveregence of a 3D Vector field'''
    wp.launch(kernel,dim = threads_shape,inputs = [current_values,dx,alpha,dimension,levels],outputs = [new_values])
    return new_values
    

def row_wise_divergence(kernel,threads_shape:tuple[int],current_values,dx:float,alpha:float,levels:wp.array[int],dimension:int,new_values:wp.array):
    wp.launch(kernel,dim = threads_shape,inputs = [current_values,dx,alpha,dimension,levels],outputs = [new_values])
    return new_values

def outer_product(kernel,
                  threads_shape,
                  vec_a:wp.array4d(dtype = wp.vec),
                  vec_b:wp.array4d(dtype = wp.vec),
                  scale:float,
                  new_values:wp.array4d(dtype = wp.mat),):
    '''Function to calculate the outerproduct of 2 vector fields'''
    wp.launch(kernel,dim = threads_shape,inputs = [vec_a,vec_b,scale],outputs = [new_values])
    return new_values
    
import warp as wp
# from ..kernels.derivatives import create_second_derivative,create_first_derivative

def second_derivative(kernel,grid,current_values,new_values,alpha):
    dimension = grid.dimension
    # num_outputs = current_values.dtype._length_
    # second_deriv = create_second_derivative(num_outputs,axis,levels)
    wp.launch(kernel,dim = grid.stencil_shape,inputs = [grid.grid,current_values,new_values,alpha,dimension])
    return new_values


    
import warp as wp
from warp.types import vector
from pde_module.FDM.utils import eligible_dims_and_shift,create_n_point_stencil_function,create_stencil_over_axis_function
from pde_module.utils.types import wp_Vector, wp_Kernel


def create_Laplacian_kernel(
    input_vector: vector, grid_shape: tuple[int, ...], stencil: vector, ghost_cells: int
) -> wp_Kernel:
    """Create a kernel for computing the Laplacian.

    Args:
        input_vector: Warp vector dtype for the input field.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the Laplacian.
    """
    assert wp.types.type_is_vector(input_vector), "Input type must be of vector"

  
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)
    dimension = len(dims)
    float_dtype = input_vector._wp_scalar_type_
    
    get_neighbors = create_n_point_stencil_function(stencil,grid_shape,input_vector)
    stencil_op = create_stencil_over_axis_function(input_vector,stencil)
    
    
    stencil_type = type(stencil)
    
    
    
    @wp.func
    def laplacian_func(stencil_array:wp.array2d(dtype = input_vector)
                       ,stencil:stencil_type,
                       alpha:float_dtype):
        laplace = input_vector()
        for axis in range(dimension):
            val_along_axis = stencil_array[axis]
            laplace += stencil_op(val_along_axis)
        return alpha*laplace
    
    @wp.kernel
    def laplacian_kernel(
        input_values: wp.array3d(dtype=input_vector),
        alpha: float_dtype,
        output_values: wp.array3d(dtype=input_vector),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        stencil_array = get_neighbors(index[0], index[1], index[2],input_values) # D,2N+1 array
        laplcian = laplacian_func(stencil_array,stencil,alpha)
        output_values[index[0], index[1], index[2]] = laplcian
        
        
    return laplacian_kernel






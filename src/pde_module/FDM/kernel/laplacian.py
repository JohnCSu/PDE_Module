import warp as wp
from warp.types import vector
from pde_module.FDM.utils import eligible_dims_and_shift
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
    stencil_type = type(stencil)
    length = stencil._length_
    num_neighbors = (length-1)//2
    
    stencil_shift = vector(length,dtype = int)(*tuple(n for n in range(-num_neighbors,num_neighbors+1)))
    
    
    @wp.func
    def laplacian_func(input_values:wp.array3d(dtype = input_vector),
                       nodeID:wp.vec3i,
                       stencil:stencil_type,
                       alpha:float_dtype):
        laplace = input_vector()
        for ii in range(dimension):
            axis = dims[ii]
            for jj in range(length):
                nodeID[axis] += stencil_shift[jj]
                laplace += stencil[jj]*input_values[nodeID[0],nodeID[1],nodeID[2]]
                nodeID[axis] -= stencil_shift[jj]
        return alpha*laplace
    
    @wp.kernel
    def laplacian_kernel(
        input_values: wp.array3d(dtype=input_vector),
        alpha: float_dtype,
        output_values: wp.array3d(dtype=input_vector),
    ):
        i, j, k = wp.tid()

        nodeID = wp.vec3i(i, j, k)
        nodeID += dims_shift
        
        output_values[nodeID[0], nodeID[1], nodeID[2]] = laplacian_func(input_values,nodeID,stencil,alpha)
        
    return laplacian_kernel






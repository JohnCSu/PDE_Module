import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from pde_module.utils.types import *
from pde_module.FDM.utils import eligible_dims_and_shift,create_n_point_stencil_function


def create_Divergence_kernel(
    input_dtype: wp_Vector | wp_Matrix,
    output_vector: vector,
    grid_shape: tuple[int, ...],
    stencil: vector,
    ghost_cells: int,
):
    """Create a kernel for computing the divergence.

    Args:
        input_dtype: Warp vector or matrix dtype for the input field.
        output_vector: Warp vector dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the divergence.
    """
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    float_dtype =  input_dtype._wp_scalar_type_
    
    divergence_op = create_divergence_op(input_dtype,stencil,grid_shape)
    
    @wp.kernel
    def Divergence_kernel(
        input_values: wp.array3d(dtype=input_dtype),
        alpha: float_dtype,
        output_values: wp.array3d(dtype=output_vector),
    ):
        i, j, k = wp.tid()

        nodeID = wp.vec3i(i, j, k)
        nodeID += dims_shift
       
        output_values[nodeID[0], nodeID[1], nodeID[2]] = divergence_op(input_values,nodeID,stencil,alpha)
        
    return Divergence_kernel




def create_divergence_op(
    input_dtype: vector, stencil: vector, grid_shape: tuple[int, ...]):
    """Create a tensor divergence operation for matrix fields.

    Args:
        input_dtype: Warp matrix dtype for the input field.
        stencil: Warp vector dtype containing stencil weights.
        grid_shape: Shape of the grid (3-tuple).
        ghost_cells: Number of ghost cells in the grid.

    Returns:
        A wp.func that computes the tensor divergence.
    """
    
    
    dims, dims_shift = eligible_dims_and_shift(grid_shape,0)
    dimension = len(dims)
    float_dtype = input_dtype._wp_scalar_type_
    stencil_type = type(stencil)
    length = stencil._length_
    num_neighbors = (length-1)//2
    
    stencil_shift = vector(length,dtype = int)(*tuple(n for n in range(-num_neighbors,num_neighbors+1)))
    
    if type_is_matrix(input_dtype):
        N,Dim  = input_dtype._shape_
        assert Dim == dimension, (
        "Dimensions of field and num col in matrix must match"
    )
        output_vec = vector(N, float_dtype)
    else:
        N  = input_dtype._length_
        assert N == dimension
        output_vec = vector(1, float_dtype)
        
    @wp.func
    def divergence_func(
        input_values:wp.array3d(dtype = input_dtype),
        nodeID:wp.vec3i,
        stencil:stencil_type,
        alpha:float_dtype #D,2N+1
        ):
        output = output_vec()
        
        for ii in range(dimension):
            axis = dims[ii]
            for jj in range(length):
                nodeID[axis] += stencil_shift[jj]
                if wp.static(type_is_matrix(input_dtype)):
                    input_val = input_values[nodeID[0],nodeID[1],nodeID[2]] # matrix
                    output += stencil[jj]*input_val[:,axis]
                else:
                    input_val = input_values[nodeID[0],nodeID[1],nodeID[2]] # Vector
                    output[0] += stencil[jj]*input_val[axis] # Can optimize this by exploiting symmetry of stencil
                nodeID[axis] -= stencil_shift[jj]

        return alpha*output

    return divergence_func










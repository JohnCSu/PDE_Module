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
    
    get_neighbors = create_n_point_stencil_function(stencil,grid_shape,input_dtype)
    divergence_op = create_divergence_op(input_dtype,stencil,grid_shape)
    
    @wp.kernel
    def Divergence_kernel(
        input_values: wp.array3d(dtype=input_dtype),
        alpha: float_dtype,
        output_values: wp.array3d(dtype=output_vector),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift
        
        stencil_array = get_neighbors(index[0], index[1], index[2],input_values) # D,2N+1 array
        divergence = divergence_op(stencil_array,stencil,alpha)
        output_values[index[0], index[1], index[2]] = divergence
        
    
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
    
    
    float_dtype =  input_dtype._wp_scalar_type_
    length = stencil._length_
    assert (length % 2) == 1, "stencil must be odd sized"
    eligible_dims, _ = eligible_dims_and_shift(grid_shape,0)    
    D = len(eligible_dims)
    
    if type_is_matrix(input_dtype):
        N,Dim  = input_dtype._shape_
        assert Dim == D, (
        "Dimensions of field and num col in matrix must match"
    )
        output_vec = vector(N, float_dtype)
    else:
        N  = input_dtype._length_
        assert N == D
        output_vec = vector(1, float_dtype)
        
    stencil_type = type(stencil)
    
    @wp.func
    def divergence_func(
        stencil_array:wp.array2d(dtype = input_dtype),
        stencil:stencil_type,
        alpha:float_dtype #D,2N+1
        ):
        output = output_vec()
        for axis in range(D): 
            # N,D we need the columns
            values = stencil_array[axis] # 2N+1 Arr
            for i in range(length): # Stencil ops 
                if wp.static(type_is_matrix(input_dtype)):
                    output += stencil[i]*values[i][:,axis] # We want to apply along columns of the input matrix
                else:    
                    output[0] += stencil[i]*values[i][axis] # We want to get a portion of each element
                
        return alpha*output

    return divergence_func










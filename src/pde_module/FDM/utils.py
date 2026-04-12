import warp as wp
from warp.types import vector
from pde_module.types import wp_Vector,wp_Matrix


def eligible_dims_and_shift(
    grid_shape: tuple[int, ...], ghost_cells: int
) -> tuple[wp_Vector, wp_Vector]:
    """Return eligible dimensions and their corresponding ghost cell shifts.

    Returns dimensions that have more than 1 point as a vector, and the
    corresponding shift for each eligible dim due to ghost_cells.

    Args:
        grid_shape: Shape of the grid (3-tuple).
        ghost_cells: Number of ghost cells.

    Returns:
        Tuple of (eligible_dims vector, shift vector).

    Example:
        (3, 5, 1) with ghost_cells > 0 returns ((0, 1), (gc, gc, 0))
    """
    d = tuple(i for i, x in enumerate(grid_shape) if x > 1)
    return wp.types.vector(length=len(d), dtype=int)(d), wp.vec3i(
        [ghost_cells if x > 1 else 0 for x in grid_shape]
    )


# def create_stencil_over_axis_function(input_dtype : wp_Vector, stencil: wp_Vector):
#     """Create a stencil operation for finite difference calculations.

#     Gathers neighbors of the current index along a given axis and applies
#     the stencil weights. Outputs a vector of the same type as input_vector.

#     Args:
#         input_dtype: Warp vector dtype for the input field.
#         stencil: Warp vector dtype containing stencil weights.
#         ghost_cells: Number of ghost cells in the grid.

#     Returns:
#         A wp.func that computes the stencil operation.
#     """
#     length = stencil._length_
#     assert (length % 2) == 1, "stencil must be odd sized"

#     @wp.func
#     def apply_stencil_over_axis(
#         values:wp.array(dtype = input_dtype), # 2N+1 array with vector/matrix dtype
#     ):
#         output = input_dtype()
#         for i in range(length):
#             output += stencil[i]*values[i]
#         return output
#     return apply_stencil_over_axis


# def create_n_point_stencil_function(stencil:wp_Vector,grid_shape:tuple[int],input_dtype:wp_Vector | wp_Matrix):
#     '''
#     Generalisation of the 'cross' point stencils e.g. Five point stenceil to n points
#     so if we have k points on either side of the central point then then we get an array of (D,2k)
#     '''
#     num_neighbors = (len(stencil)-1)//2

#     dims,_ = eligible_dims_and_shift(grid_shape,0) # We just need the availiable dims
#     dimension = len(dims)
    
#     shape = vector(dimension,dtype = int)(tuple(num_neighbors*2+1  for _ in dims))
    
#     shift_mapping = vector(num_neighbors*2+1,dtype = int)(tuple(i for i in range(-num_neighbors, num_neighbors + 1)))
    
#     array_shape = (dimension,num_neighbors*2+1)
    
#     @wp.func
#     def get_neighbors(i:int,j:int,k:int,arr:wp.array3d(dtype = input_dtype)):
#         tmp = wp.zeros(shape = (wp.static(array_shape[0]),wp.static(array_shape[1])),dtype=input_dtype)
#         shift = wp.vec3i()
        
#         for ii in range(wp.static(dimension)): # Axis
#             for n in range(wp.static(shape[ii])): # num neighbors in each
#                 axis = dims[ii] # Get the corresponding axis
#                 shift[axis] = shift_mapping[n]
#                 tmp[ii,n] = arr[i+shift[0],j+shift[1],k+shift[2]]
#                 shift[axis] = 0 # Reset Index shift
        
#         return tmp
        
#     return get_neighbors
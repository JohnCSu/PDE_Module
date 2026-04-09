import warp as wp
from warp.types import vector, type_is_vector, type_is_matrix
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Kernel

from pde_module.FDM.utils import eligible_dims_and_shift,create_n_point_stencil_function,create_stencil_over_axis_function

def create_Grad_kernel(
    input_vector: wp_Vector,
    output_dtype: wp_Vector | wp_Matrix,
    grid_shape: tuple[int, ...],
    stencil: wp_Vector,
    ghost_cells: int,
) -> wp_Kernel:
    """Create a kernel for computing the gradient.

    Args:
        input_vector: Warp vector dtype for the input field.
        output_dtype: Warp vector or matrix dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the gradient.
    """
    assert type_is_vector(input_vector)
    assert type_is_vector(output_dtype) or type_is_matrix(output_dtype)
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    float_dtype =  input_vector._wp_scalar_type_
    
    get_neighbors = create_n_point_stencil_function(stencil,grid_shape,input_vector)
    grad_op = create_grad_op(input_vector,output_dtype,stencil,grid_shape)
    
    
    @wp.kernel
    def grad_kernel(
        input_values: wp.array3d(dtype=input_vector),
        alpha: float_dtype,
        output_values: wp.array3d(dtype=output_dtype),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift
        
        stencil_array = get_neighbors(index[0], index[1], index[2],input_values) # D,2N+1 array
        grad = grad_op(stencil_array,stencil,alpha)
        output_values[index[0], index[1], index[2]] = grad
    
    
    
    return grad_kernel


def create_grad_op( input_dtype: vector,output_dtype:wp_Matrix|wp_Vector, stencil: vector, grid_shape: tuple[int, ...]):
    float_dtype =  input_dtype._wp_scalar_type_
    length = stencil._length_
    assert (length % 2) == 1, "stencil must be odd sized"
    eligible_dims, _ = eligible_dims_and_shift(grid_shape,0)    
    D = len(eligible_dims)
    stencil_type = type(stencil)
    
    if type_is_vector(output_dtype):
        assert input_dtype._length_ == 1, 'Grad Kernel with Vector output only availiable for vectors of length 1'
        assert output_dtype._length_ == D
    else:
        N,Dim  = output_dtype._shape_
        assert Dim == D 
        assert N == input_dtype._length_
    
    @wp.func
    def grad_func(
        stencil_array:wp.array2d(dtype = input_dtype),
        stencil:stencil_type,
        alpha:float_dtype #D,2N+1
        ):
        output = output_dtype()
        for axis in range(D): 
            # N,D we need the columns
            values = stencil_array[axis] # 2N+1 Arr
            for i in range(length): # Stencil ops 
                if wp.static(type_is_vector(output_dtype)):
                    output[axis] += stencil[i]*values[i][0] # Float * Vector(Length=1) 
                else:
                    output[:,axis] += stencil[i]*values[i] # Float * Vector(Length=N)
        
        return alpha*output
            
    return grad_func
    
    
    
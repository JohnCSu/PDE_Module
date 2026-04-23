import warp as wp
from warp.types import vector, type_is_vector, type_is_matrix
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils.types import wp_Vector, wp_Matrix, wp_Kernel

from pde_module.FDM.utils import eligible_dims_and_shift

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

        
        nodeID = wp.vec3i(i, j, k)
        nodeID += dims_shift
       
        output_values[nodeID[0], nodeID[1], nodeID[2]] = grad_op(input_values,nodeID,stencil,alpha)
        
    
    
    return grad_kernel


def create_grad_op( input_dtype: vector,output_dtype:wp_Matrix|wp_Vector, stencil: vector, grid_shape: tuple[int, ...]):
    
    dims, dims_shift = eligible_dims_and_shift(grid_shape,0)
    dimension = len(dims)
    float_dtype = input_dtype._wp_scalar_type_
    stencil_type = type(stencil)
    length = stencil._length_
    num_neighbors = (length-1)//2
    
    stencil_shift = vector(length,dtype = int)(*tuple(n for n in range(-num_neighbors,num_neighbors+1)))
    
    assert (length % 2) == 1, "stencil must be odd sized"
    
    
    if type_is_vector(output_dtype):
        assert input_dtype._length_ == 1, 'Grad Kernel with Vector output only availiable for vectors of length 1'
        assert output_dtype._length_ == dimension
    else:
        N,Dim  = output_dtype._shape_
        assert Dim == dimension 
        assert N == input_dtype._length_
    
    @wp.func
    def grad_func(
        input_values:wp.array3d(dtype = input_dtype),
        nodeID:wp.vec3i,
        stencil:stencil_type,
        alpha:float_dtype #D,2N+1
        ):
        output = output_dtype()
        
        for ii in range(dimension):
            axis = dims[ii]
            for jj in range(length):
                nodeID[axis] += stencil_shift[jj]
                if wp.static(type_is_vector(output_dtype)):
                    input_val = input_values[nodeID[0],nodeID[1],nodeID[2]][0] # Vector(Length=1) -> Vector(L = dimension)
                    output[axis] += stencil[jj]*input_val
                else:
                    input_val = input_values[nodeID[0],nodeID[1],nodeID[2]] # Vector(L = N) -> Matrix(L,D)
                    output[:,axis] += stencil[jj]*input_val
            
                nodeID[axis] -= stencil_shift[jj]
        
        return alpha*output
            
    return grad_func
    
    
    
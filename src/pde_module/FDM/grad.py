from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from ..stencil.hooks import *
from pde_module.stencil.utils import create_stencil_op,eligible_dims_and_shift

class Grad(ExplicitUniformGridStencil):
    '''
    Create Grad Stencil of vector field Using Central Based Finite difference.
    
    If inputs is equal to 1, the default is to set output to a vector, use force_matrix to set output dtype as a matrix instead
    
    Args
    ----------
    inputs : int 
        length of vector
    grid_shape: tuple[int]
        grid shape, should have length of 3 (with 1 indicating it is not a valid dimension)
    dx : float 
        grid spacing
    ghost_cells : int 
        number of ghost cells on the grid
    stencil : vector | None
        stencil to use for grad. If None, 2nd Order stencil is used
        
    '''
    def __init__(self,inputs:int,grid_shape:tuple[int],dx:float,ghost_cells,stencil = None,force_matrix:bool = False, float_dtype=wp.float32):
        
        self.force_matrix = force_matrix
        dimension = self.calculate_dimension_from_grid_shape(grid_shape)
        self.stencil = wp.types.vector(3,dtype = float_dtype)([-1./(2*dx),0.,1/(2*dx)])
        
        assert type(inputs) is int and inputs > 0
        
        if inputs == 1 and force_matrix is False:
            output_shape = dimension
        else:
            output_shape = (inputs,dimension)
        
        assert dimension <= 3
        
        super().__init__(inputs,output_shape,dx,ghost_cells,float_dtype)
        
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert types_equal(self.input_dtype,input_array.dtype)
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        self.kernel = create_Grad_kernel(self.input_dtype,self.output_dtype,input_array.shape,self.stencil,self.ghost_cells)
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    def forward(self, input_array,alpha = 1.):
        '''
        Args
        ---------
            input_array : wp.array3d 
                A 3D array with vector of size N to calculate grad from
            alpha : float
                proportionality term to scale the laplacian term. Default is 1.
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array where each element is a matrix or vector.
                - If vector length (N) == 1: return vector of the dimension (D) of the grid or a (1,D) matrix if force_matrix = True
                - If vector length (N) > 1: return matrix of size (N,D)
        '''
            
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    


def create_Grad_kernel(input_vector:vector,output_dtype,grid_shape,stencil,ghost_cells):
    assert type_is_vector(input_vector)
    
    output_dtype_is_vector =  type_is_vector(output_dtype)
    assert type_is_vector(output_dtype) or type_is_matrix(output_dtype)
    
    
    stencil_op = create_stencil_op(input_vector,stencil,ghost_cells)
    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells)
    @wp.kernel
    def grad_kernel(
        scalar_array:wp.array3d(dtype = input_vector),
        alpha:input_vector._wp_scalar_type_,
        grad_array:wp.array3d(dtype = output_dtype), 
    ):
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += dims_shift
        
        
        grad = output_dtype()
        
        for i in range(wp.static(len(dims))):
            if wp.static(output_dtype_is_vector):
                grad[i] = stencil_op(scalar_array,index,stencil,dims[i])[0] # Scalar value
            else:
                grad[:,i] = stencil_op(scalar_array,index,stencil,dims[i]) # Vector of derivatives of all outputs wrt to some axis

        grad *= alpha
        
        grad_array[index[0],index[1],index[2]] = grad
    return grad_kernel
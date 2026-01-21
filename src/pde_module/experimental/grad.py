from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix
from .hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift


class Grad(ExplicitUniformGridStencil):
    '''
    Calculate gradient of a scalar vector or the jacobian of a vector
    '''
    def __init__(self,field,dx:float,ghost_cells, float_dtype=wp.float32,force_matrix:bool = False):
        assert type_is_vector(field)
        
        self.force_matrix = force_matrix
        dimension = sum([1 if g > 1 else 0 for g in field.shape])
        
        if field.dtype._length_ == 1 and force_matrix is False:
            output_shape = dimension
        else:
            output_shape = (field.dtype._length,dimension)
        
        
        assert dimension <= 3
        
        super().__init__(1,output_shape,dx,ghost_cells,float_dtype)
        
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        self.kernel = create_Grad_kernel(self.input_dtype,input_array.shape,self.stencil)
        self.kernel_dim = self.field_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    def forward(self, input_array,alpha = 1.,*args,**kwargs):    
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    


def create_Grad_kernel(input_vector:vector,output_dtype,grid_shape,stencil,ghost_cells):
    assert type_is_vector(input_vector)
    
    output_type = 'vector' if hasattr(output_dtype._length_) else 'matrix'
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
            if wp.static(output_type == 'vector'):
                grad[i] = stencil_op(scalar_array,index,stencil,dims[i])[0] # Scalar value
            else:
                grad[:,i] = stencil_op(scalar_array,index,stencil,dims[i]) # Vector of derivatives of all outputs wrt to some axis

        grad *= alpha
        
        grad_array[index[0],index[1],index[2]] = grad
    return grad_kernel
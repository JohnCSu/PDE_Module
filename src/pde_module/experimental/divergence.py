from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix
from .hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift



class Divergence(ExplicitUniformGridStencil):
    '''
    Calculate Divergence of a vector field to a scalar field (vector field of size 1)
    '''
    def __init__(self, field,dx:float,stencil = None, float_dtype=wp.float32):
        
        if stencil is None:
            self.stencil = wp.types.vector(3,dtype = float_dtype)([-1./(2*dx),0.,1/(2*dx)])
        else:
            raise ValueError('Custom stencil not implemented yet')        
        assert (self.stencil._length_ % 2) == 1,'stencil must be odd sized'

        ghost_cells = (self.stencil._length_ -1)// 2

        dimension = self.calculate_dimension_from_field_shape(field.shape)
        
        super().__init__(dimension, 1,dx,ghost_cells,float_dtype=float_dtype)
        
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        self.kernel = create_Divergence_kernel(self.input_dtype,self.output_dtype,input_array.shape,self.stencil,self.ghost_cells)
        self.kernel_dim = self.field_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    def forward(self, input_array,alpha = 1.,*args,**kwargs):    
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    
    
def create_Divergence_kernel(input_vector,output_vector,grid_shape,stencil,ghost_cells):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    
    assert wp.types.type_is_vector(input_vector), 'Input type must be of vector'
    
    stencil_op = create_stencil_op(input_vector,stencil,ghost_cells)
    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells) 
    
    assert output_vector._length_ == 1
    
    @wp.kernel
    def Divergence_kernel(input_values:wp.array3d(dtype = input_vector),
                        alpha:input_vector._wp_scalar_type_,
                        output_values:wp.array3d(dtype = output_vector),):
        
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += dims_shift
        
        div = output_vector() # Vector same length as input array vec
        # Can do more effecient on per val but for now lets resues stencil op
        for i in range(wp.static(len(dims))):
            div[0] += stencil_op(input_values,index,stencil,dims[i])[i]    
        
        div *= alpha
        
        output_values[index[0],index[1],index[2]] = div

    return Divergence_kernel

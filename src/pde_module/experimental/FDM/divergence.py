from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from ..Stencil.hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift



class Divergence(ExplicitUniformGridStencil):
    '''
    Calculate Divergence of a vector field
    '''
    def __init__(self,div_type,field_shape,dx:float,ghost_cells,stencil = None, float_dtype=wp.float32):
        
        if stencil is None:
            self.stencil = wp.types.vector(3,dtype = float_dtype)([-1./(2*dx),0.,1/(2*dx)])
        else:
            raise ValueError('Custom stencil not implemented yet')        
        assert (self.stencil._length_ % 2) == 1,'stencil must be odd sized'

        if div_type == 'vector':
            dimension = self.calculate_dimension_from_field_shape(field_shape)
            inputs = dimension
            outputs = 1
            super().__init__(inputs, outputs,dx,ghost_cells,float_dtype=float_dtype)
            self.div_type = 'vector'
            
        elif div_type == 'tensor':
            self.div_type =='tensor'
            self._temp_args = [dx,ghost_cells,float_dtype]
        else:
            raise ValueError(f'Valid div_type is strings "vector" or "tensor"')
        
    
    @property
    def valid_div_types(self):
        return {'vector','tensor'}
    
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert type_is_vector(input_array.dtype)
        
        if self.div_type == 'tensor':
            inputs = input_array.dtype._length_
            outputs = inputs
            super().__init__(inputs, outputs,*self._temp_args)    
        
        # This checks if the input array matches the original dtype for vector based divergence
        assert types_equal(self.input_dtype,input_array.dtype)
        
        self.kernel = create_Divergence_kernel(self.input_dtype,self.output_dtype,input_array.shape,self.stencil,self.ghost_cells,self.div_type)
        self.kernel_dim = self.field_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    def forward(self, input_array,alpha = 1.,*args,**kwargs):    
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    
    
def create_Divergence_kernel(input_vector,output_vector,grid_shape,stencil,ghost_cells,div_type):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    
    
    assert wp.types.type_is_vector(input_vector), 'Input type must be of vector'
    
    stencil_op = create_stencil_op(input_vector,stencil,ghost_cells)
    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells) 
    
    assert div_type in {'vector','tensor'}
    
    if div_type == 'vector':
        assert output_vector._length_ == 1
    else:
        assert types_equal(input_vector,output_vector)
    
    
    @wp.kernel
    def Divergence_kernel(input_values:wp.array3d(dtype = input_vector),
                        alpha:input_vector._wp_scalar_type_,
                        output_values:wp.array3d(dtype = output_vector),):
        
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += dims_shift
        
        div = output_vector() # Vector same length as input array vec
        if wp.static(div_type == 'tensor'):
            for i in range(wp.static(len(dims))):
                div += stencil_op(input_values,index,stencil,dims[i])
        else:# Can do more effecient on per val but for now lets resues stencil op
            for i in range(wp.static(len(dims))):
                div[0] += stencil_op(input_values,index,stencil,dims[i])[i]    
            
        div *= alpha
            
        output_values[index[0],index[1],index[2]] = div

    return Divergence_kernel

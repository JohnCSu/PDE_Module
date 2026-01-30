from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from ..Stencil.hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift,create_tensor_divergence_op

from collections.abc import Iterable

class Divergence(ExplicitUniformGridStencil):
    '''
    Calculate Divergence of a vector field of size D or matrix of shape N,D where D is the dimension of the grid
    
    div_type: vector or tensor
    '''
    def __init__(self,inputs:int | tuple,grid_shape,dx:float,ghost_cells,stencil = None, float_dtype=wp.float32):
        
        if stencil is None:
            self.stencil = wp.types.vector(3,dtype = float_dtype)([-1./(2*dx),0.,1/(2*dx)])
        else:
            raise ValueError('Custom stencil not implemented yet')        
        assert (self.stencil._length_ % 2) == 1,'stencil must be odd sized'
        
        dimension = self.calculate_dimension_from_grid_shape(grid_shape)
        
        
        # Determine if vector or matrix inputs
        is_vector = False
        if isinstance(inputs,Iterable):
            assert all(type(i) is int for i in inputs)
            if len(inputs) == 1:
                inputs = inputs[0]
                is_vector = True
            else:
                assert len(inputs) == 2
        elif type(inputs) is int:
            is_vector = True
        else:
            raise ValueError('Inputs must be int or tuple|list of int')
        
        if is_vector:
            assert inputs == dimension
            outputs = 1
        else:
            assert inputs[1] == dimension
            outputs = inputs[0]
            
        super().__init__(inputs, outputs,dx,ghost_cells,float_dtype=float_dtype)
        
    
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert types_equal(self.input_dtype,input_array.dtype)
        
        self.kernel = create_Divergence_kernel(self.input_dtype,self.output_dtype,input_array.shape,self.stencil,self.ghost_cells)
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    def forward(self, input_array,alpha = 1.,*args,**kwargs):    
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    
    
def create_Divergence_kernel(input_dtype,output_vector,grid_shape,stencil,ghost_cells):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells) 
    
    # assert div_type in {'vector','tensor'}
    
    if type_is_vector(input_dtype):
        div_op = create_stencil_op(input_dtype,stencil,ghost_cells)
    else:
        div_op = create_tensor_divergence_op(input_dtype,stencil,grid_shape,ghost_cells)
    
    @wp.kernel
    def Divergence_kernel(input_values:wp.array3d(dtype = input_dtype),
                        alpha:input_dtype._wp_scalar_type_,
                        output_values:wp.array3d(dtype = output_vector),):
        
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += dims_shift
        
        div = output_vector() # Vector same length as input array vec
        if wp.static(type_is_matrix(input_dtype)):
            div = div_op(input_values,index,stencil)
        else:# Can do more effecient on per val but for now lets resues stencil op
            for i in range(wp.static(len(dims))):
                div[0] += div_op(input_values,index,stencil,dims[i])[i]    
            
        div *= alpha
            
        output_values[index[0],index[1],index[2]] = div

    return Divergence_kernel

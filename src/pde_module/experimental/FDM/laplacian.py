# from .stencil import Stencil
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
# from .types import *
from ..Stencil.hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift
import numpy as np


np.concat
class Laplacian(ExplicitUniformGridStencil):
    '''
    Create Laplacian Stencil of vector field Using Central Based Finite difference.
    
    Args
    ----------
    inputs : int 
        length of vector
    dx : float 
        grid spacing
    ghost_cells : int 
        number of ghost cells on the grid
    stencil : vector | None
        stencil to use for laplacian. If None, 2nd Order stencil is used
        
    '''
    def __init__(self,inputs:int,dx:float,ghost_cells,stencil = None, float_dtype=wp.float32):
        
        if stencil is None:
            self.stencil = wp.types.vector(3,dtype = float_dtype)([1./dx**2,-2./dx**2,1/dx**2])
        else:
            raise ValueError('Custom stencil not implemented yet')        
        assert (self.stencil._length_ % 2) == 1,'stencil must be odd sized'
        
        super().__init__( inputs,inputs,dx,ghost_cells,float_dtype=float_dtype)
        
    @setup
    def initialize_kernel(self,input_array,*args, **kwargs):
        assert types_equal(self.input_dtype,input_array.dtype)
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        self.kernel = create_Laplacian_kernel(self.input_dtype,input_array.shape,self.stencil,self.ghost_cells)
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(input_array.shape,self.ghost_cells)
    
    
    def forward(self, input_array,alpha = 1.,*args,**kwargs):
        '''
        Args
        ---------
            input_array : wp.array3d 
                A 3D array with vector dtype to calculate laplacian from
            alpha : float
                proportionality term to scale the laplacian term. Default is 1.
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array with same vector dtype as input_array representing laplacian of each term
        '''
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array
    


def create_Laplacian_kernel(input_vector,grid_shape,stencil,ghost_cells):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    
    assert wp.types.type_is_vector(input_vector), 'Input type must be of vector'
    
    stencil_op = create_stencil_op(input_vector,stencil,ghost_cells)
    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells) 
    
    @wp.kernel
    def laplacian_kernel(
                        input_values:wp.array3d(dtype = input_vector),
                        alpha:input_vector._wp_scalar_type_,
                        output_values:wp.array3d(dtype = input_vector),
                        ):
        
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += dims_shift
        
        laplace = input_vector() # Vector same length as input array vec
        for i in range(wp.static(len(dims))):
            laplace += stencil_op(input_values,index,stencil,dims[i])    
        
        laplace *= alpha
        
        output_values[index[0],index[1],index[2]] = laplace
        
    return laplacian_kernel
        
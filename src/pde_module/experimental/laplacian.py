# from .stencil import Stencil
from .uniformGridStencil import UniformGridStencil
import warp as wp
from warp.types import vector,matrix
# from .types import *

class Laplacian(UniformGridStencil):
    '''
    Calculate Divergence using uniform central differences
    '''
    def __init__(self, inputs:int,dx:float,stencil = None, float_dtype=wp.float32):
        assert isinstance(inputs,int)
        super().__init__(inputs, inputs,dx,float_dtype)
        
        if stencil is None:
            self.stencil = wp.types.vector(3,dtype = self.float_dtype)([1./dx**2,-2./dx**2,1/dx**2])
        else:
            raise ValueError('Custom stencil not implemented yet')        
        assert (self.stencil._length_ % 2) == 1,'stencil must be odd sized'

    def initialize_kernel(self,input_array,*args, **kwargs):
        assert len(self.inputs) == 1,'Laplacian Only For Vectors'
        self.kernel = create_Laplacian_kernel(self.input_dtype,input_array.shape,self.stencil,self.float_dtype)
        self.kernel_dim = self.get_ghost_shape(input_array.shape,self.stencil)
    
    def forward(self, input_array,alpha = 1):
        
        wp.launch(self.kernel,dim = self.kernel_dim,inputs = [
            input_array,
            alpha,   
        ],
        outputs= [self.output_array])
        return self.output_array




def create_stencil_op(input_vector:vector,stencil:vector,float_dtype):
    length = stencil._length_
    assert (length % 2) == 1,'stencil must be odd sized'
    num_ghost_cells = (length -1)//2
    

    @wp.func
    def stencil_op(current_values:wp.array3d(dtype=input_vector),
                   index:wp.vec3i,
                   stencil:stencil,
                   axis:int):
        
        value = input_vector()
        for i in range(wp.static(length)):
            shift = i - num_ghost_cells
            stencil_index = index
            stencil_index[axis] = index[axis] + shift
            value += current_values[stencil_index[0],stencil_index[1],stencil_index[2]]*stencil[i]

        return value
            
    return stencil_op


def create_Laplacian_kernel(input_vector,grid_shape,stencil,float_dtype):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    
    assert wp.types.type_is_vector(input_vector), 'Input type must be of vector'
    
    d = [i for i,x in enumerate(grid_shape) if x > 1 ] # Store dimensions that are eligible
    
    axes = wp.types.vector(length=len(d),dtype = int)(d)
    
    length = stencil._length_
    assert (length % 2) == 1,'stencil must be odd sized'
    num_ghost_cells = (length -1)//2
    
    stencil_op = create_stencil_op(input_vector,stencil,float_dtype)
    axes_shift = wp.vec3i([num_ghost_cells if x > 1 else 0 for x in grid_shape])
    
    @wp.kernel
    def laplacian_kernel(
                        current_values:wp.array3d(dtype = input_vector),
                        alpha:float_dtype,
                        new_values:wp.array3d(dtype = input_vector),
                        ):
        
        i,j,k = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        index = wp.vec3i(i,j,k) 
        index += axes_shift
        
        laplace = input_vector() # Vector same length as input array vec
        for i in range(wp.static(len(d))):
            laplace += stencil_op(current_values,index,stencil,axes[i])    
        
        new_values[index[0],index[1],index[2]] = alpha*laplace
        
    return laplacian_kernel
        
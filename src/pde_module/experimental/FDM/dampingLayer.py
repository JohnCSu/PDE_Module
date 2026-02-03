import warp as wp
import numpy as np
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from warp.types import vector,matrix,type_is_vector,type_is_matrix,types_equal
from ..Stencil.hooks import *
from pde_module.experimental.stencil_utils import create_stencil_op,eligible_dims_and_shift,create_tensor_divergence_op
from pde_module.experimental.constants import INT32_MAX
from collections.abc import Iterable


class DampingLayer(ExplicitUniformGridStencil):
    '''
    Apply a damping layer around the grid
    
    Calculates:
    -damp_factor*(input_array[grid_point] - farfield_condition)
    
    '''
    def __init__(self,inputs:int | tuple,num_layers,beta,grid_shape,dx:float,ghost_cells, float_dtype=wp.float32):
        super().__init__(inputs, inputs,dx,ghost_cells,float_dtype=float_dtype)
        self.grid_shape = grid_shape
        self.num_layers = num_layers
        self.beta = beta
        self.outer_layers = self.get_outer_grid_points(num_layers,self.grid_shape)
        
    @staticmethod
    def get_outer_grid_points(num_layers,grid_shape):
        indices = np.indices(grid_shape,dtype = np.int32)
        indices = np.moveaxis(indices,0,-1).reshape(-1,3)
        
        masks = []
        for i,s in enumerate(grid_shape):
            if s> 1:
                a1 = indices[:,i] < num_layers
                a2 = indices[:,i] >= (s-num_layers)
                
                masks.append(np.logical_or(a1,a2))
        
        
        # masks = np.logical_or(*masks)
        
        mask = masks[0]
        for m in masks[1:]:
            mask = np.logical_or(mask,m)
        
        outer_layers = indices[mask]
        return outer_layers
        
    @setup
    def initialize_kernel(self, input_array, *args, **kwargs):
        self.warp_outer_points = wp.array(self.outer_layers,dtype = wp.vec3i)
        self.kernel = create_DampingLayer_kernel(self.input_dtype,self.num_layers,self.beta,self.grid_shape,self.ghost_cells)
    
    @setup
    def zero_array(self,*args,**kwargs):
        self.output_array.zero_()
    
    def forward(self,input_array,farfield_condition,sigma_max,*args, **kwargs):
        wp.launch(self.kernel,dim = len(self.warp_outer_points),inputs = [input_array,self.warp_outer_points,farfield_condition,sigma_max],outputs=[self.output_array])
        return self.output_array
        
    

def create_DampingLayer_kernel(input_vector,num_layers,beta,grid_shape,ghost_cells):
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    dimension = len(eligible_dims)
    limits = matrix(shape = (3,2), dtype= int)()
    
    i = 0 
    for i,s in enumerate(grid_shape):
        if s > 1:
        # -1 for End of axis to account for the fact indexing starts at 0 and ends n-1
            limits[i] = wp.vec2i([num_layers,(s-1) - num_layers ])
        
    
        
    
    float_type = input_vector._wp_scalar_type_
    grid_shape_vec = wp.vec3i(grid_shape)
    
    @wp.func
    def get_argmin_and_min(grid_point:wp.vec3i):
        
        out = wp.vec3i()
        
        for i in range(3): 
            boundary = grid_shape_vec[i]
            point = grid_point[i]
            out[i] = wp.where(boundary > 1 ,wp.min(point,boundary - point - 1),INT32_MAX) # Need the -1 to account boundary is a count
        # return wp.min(out),wp.argmin(out)
        return wp.int32(wp.argmin(out))
    
    @wp.kernel
    def DampingLayer(
        input_array:wp.array3d(dtype = input_vector),
        grid_points:wp.array(dtype = wp.vec3i),
        farfield_condition : input_vector,
        sigma_max:float_type,
        output_array:wp.array3d(dtype = input_vector), 
    ):
        tid = wp.tid()
        
        grid_point = grid_points[tid]
        
        axis = get_argmin_and_min(grid_point)
        
        #  = wp.int32(wp.argmin(out))
        assert 0 <= axis < 2
        
        x = grid_point[axis]
        if (grid_shape_vec[axis]- 1 - x ) < x : # We have a min point on the L end
            dx = x-limits[axis,1]
        else:
            dx = limits[axis,0]-x
        # For LHS
        damp_factor = sigma_max*(float_type(dx)/float_type(num_layers))**beta
        # # For RHS
        output_array[grid_point[0],grid_point[1],grid_point[2]] = -damp_factor*(input_array[grid_point[0],grid_point[1],grid_point[2]] - farfield_condition) #
        
    return DampingLayer                                          
        
        
    
    
    
        
        
        
import warp as wp
import numpy as np
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from warp.types import vector,matrix
from ..stencil.hooks import *
from pde_module.stencil.utils import eligible_dims_and_shift
from pde_module.utils.constants import INT32_MAX
from collections.abc import Iterable


class ViscousDampingLayer(ExplicitUniformGridStencil):
    '''
    Apply a damping layer num_layers thick proportional to the time derivative of your field around the outer layers of the grid
    
    Args
    ----------
    inputs : int | tuple[int]
        input shape: 
            - If int, then input is a vector is assumed. vector length must be same as grid dimension
            - If tuple[int] then matrix is assumed. Number of columns must be same as grid dimension
    grid_shape: tuple[int]
        grid shape, should have length of 3 (with 1 indicating it is not a valid dimension)
    dx : float 
        grid spacing
    ghost_cells : int 
        number of ghost cells on the grid
    p:int
        polynomial order of damping stregnth. Default is 2
    float_dtype : wp.float32 | wp.float64
        float type. default is wp.float32
    
    Note the sign of the output is already -ve:
    
    -sigma*(input_array[grid_point] - farfield_condition)
    
    '''
    def __init__(self,inputs:int | tuple,num_layers,grid_shape,dx:float,ghost_cells,p=2, float_dtype=wp.float32):
        super().__init__(inputs, inputs,dx,ghost_cells,float_dtype=float_dtype)
        self.grid_shape = grid_shape
        self.num_layers = num_layers
        self.p = self.float_dtype(p)
        self.groups = {}
        self.exclude_group = set()
        self.get_damping_grid_points(num_layers,grid_shape)
        # self.outer_layers = self.get_outer_grid_points(num_layers,self.grid_shape)
    def exclude(self,*groups):
        '''
        Groups that are excludied from damping zone e.g inlet
        '''
        for group in groups:
            assert group in self.groups.keys()
            self.exclude_group.add(group)
            
            
    def get_damping_grid_points(self,num_layers,grid_shape):
        
        self.indices = np.indices(grid_shape,dtype = np.int32)
        self.indices = np.moveaxis(self.indices,0,-1).reshape(-1,3)
        
        for i,(s,axis_name) in enumerate(zip(grid_shape,['X','Y','Z'])):
            if s > 1:
                a1 = self.indices[:,i] < num_layers
                a2 = self.indices[:,i] >= (s-num_layers)
                
                self.groups[f'-{axis_name}'] = a1
                self.groups[f'+{axis_name}'] = a2     
    
    def get_damping_array(self):
        masks = [val for key,val in self.groups.items() if key not in self.exclude_group]
        mask = masks[0]
        for m in masks[1:]:
            mask = np.logical_or(mask,m)
                
        outer_layers = self.indices[mask]
        return outer_layers
        
    @staticmethod
    def calculate_C_max(c:float,L:float,p:float = 2,R:float = 1e-2):
        return float((p+1.)*c/(2.*L)*np.log(1./R))
    
    
    def __call__(self, du_dt,C_max):
        '''
        Args
        ---------
            du_dt : wp.array3d 
                A 3D array with that matches the input shape representing the time derivative of each variable
            C_max : float
                proportionality term to scale the damping layer. Recommended is 
                (p+1)c/(2*L) * ln(1/R) where:
                - p: polynomial order
                - c: wave speed
                - L: thickness of damping layer (in distance units not number of cells)
                - R: Target reflection (typically ~ 1e-2 - 1e-6)
                
                To ensure stability aim for C_max*dt <= 0.5, if C_max*dt >= 1. the simulation will blow up
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array with same shape and dtype as the input_array
        '''    
        return super().__call__(du_dt,C_max)
    
    @setup
    def initialize_kernel(self,du_dt,C_max):
        self.outer_layers = self.get_damping_array()
        self.warp_outer_points = wp.array(self.outer_layers,dtype = wp.vec3i)
        self.kernel = create_dampingLayer_kernel(self.input_dtype,self.num_layers,self.p,self.grid_shape,self.ghost_cells)
        
    @setup
    def zero_array(self,*args,**kwargs):
        self.output_array.zero_()
    
    def forward(self,du_dt,C_max):
        '''
        Args
        ---------
            input_array : wp.array3d 
                A 3D array with that matches the input shape (either vector or matrix)
            farfield_condition: wp.vector | wp.matrix
                farfield condition the damping layers force the solution to match
            C_max : float
                proportionality term to scale the damping layer. Recommended is alpha/dt where alpha ~ 0.05 - 0.5
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array with same shape and dtype as the input_array
        '''    
        wp.launch(self.kernel,dim = len(self.warp_outer_points),inputs = [du_dt,self.warp_outer_points,C_max],outputs=[self.output_array])
        return self.output_array
        
    

def create_dampingLayer_kernel(input_dtype ,num_layers,p,grid_shape,ghost_cells):
    eligible_dims,_ = eligible_dims_and_shift(grid_shape,ghost_cells)
    dimension = len(eligible_dims)
    limits = matrix(shape = (3,2), dtype= int)()
    
    i = 0 
    for i,s in enumerate(grid_shape):
        if s > 1:
        # -1 for End of axis to account for the fact indexing starts at 0 and ends n-1
            limits[i] = wp.vec2i([num_layers,(s-1) - num_layers ])

    float_type = input_dtype ._wp_scalar_type_
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
    def dampingLayer(
        du_dt:wp.array3d(dtype = input_dtype),
        grid_points:wp.array(dtype = wp.vec3i),
        C_max:float_type,
        output_array:wp.array3d(dtype = input_dtype ), 
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
        damp_factor = C_max*(float_type(dx)/float_type(num_layers))**p
        # # For RHS
        output_array[grid_point[0],grid_point[1],grid_point[2]] = -damp_factor*du_dt[grid_point[0],grid_point[1],grid_point[2]]#
        
    return dampingLayer                                          
        
        
    
    
    
        
        
        
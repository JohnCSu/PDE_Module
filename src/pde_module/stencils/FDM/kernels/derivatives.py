from . import second_order,first_order
from .neighbors import get_adjacent_points_along_axis_function,get_adjacent_values_along_axis
import warp as wp

# '''
# Stencil Functions for Second Order gradients and 2nd derivatives
# '''


def create_first_derivative_kernel(num_outputs,axis,levels):
    '''First Order Derivative in axis direction using second order (for now)
    If include_boundary is set to true then we also calculate the gradient at the boundary 
    '''
    # dimension = grid.dimension
    # num_outputs = current_values.dtype._length_
    get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    get_adjacent_values = get_adjacent_values_along_axis(num_outputs,levels)
    
    @wp.kernel
    def dfdx(grid_points:wp.array3d(dtype = wp.vec(length=3,dtype=float)),
                        current_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        new_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        alpha:float,
                        dimension:int):
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust fro stencil shape
        x_id += 1
        
        if dimension >= 2:
            y_id += 1
        
        if dimension == 3:
            z_id += 1
        
        # Step 2. Get current indexed grid point and value
        current_point = grid_points[x_id,y_id,z_id]
        current_value = current_values[batch_id,x_id,y_id,z_id]
        
        # for axis in range(wp.static(dimension)):
        adj_points = get_adjacent_points_along_axis(grid_points,x_id,y_id,z_id,axis)
        adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
        new_values[batch_id,x_id,y_id,z_id] = alpha*first_order.central_difference(adj_points[1],current_point,adj_points[0], adj_values[1],current_value,adj_values[0])
        
    return dfdx
        
    
def create_second_derivative_kernel(axis,num_outputs,levels):
    '''First Order Derivative in axis direction using second order (for now)
    If include_boundary is set to true then we also calculate the gradient at the boundary 
    '''
    
    get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    get_adjacent_values = get_adjacent_values_along_axis(num_outputs,levels)
    
    
    @wp.kernel
    def d2fdx2(grid_points:wp.array3d(dtype = wp.vec(length=3,dtype=float)),
                        current_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        new_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        alpha:float,
                        dimension:int):
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust fro stencil shape
        x_id += 1
        
        if dimension >= 2:
            y_id += 1
        
        if dimension == 3:
            z_id += 1
        
        # Step 2. Get current indexed grid point and value
        current_point = grid_points[x_id,y_id,z_id]
        current_value = current_values[batch_id,x_id,y_id,z_id]
        
        # for axis in range(wp.static(dimension)):
        adj_points = get_adjacent_points_along_axis(grid_points,x_id,y_id,z_id,axis)
        adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
        new_values[batch_id,x_id,y_id,z_id] = alpha*second_order.central_difference(adj_points[1],current_point,adj_points[0], adj_values[1],current_value,adj_values[0])
        
    
    return d2fdx2
        
    
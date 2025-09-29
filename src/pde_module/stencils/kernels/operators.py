from . import second_order
from . import first_order
from .neighbors import get_adjacent_values_along_axis,get_adjacent_points
import warp as wp


def create_Laplacian_kernel(dimension,num_inputs,levels):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    get_adjacent_points_along_axis = get_adjacent_points(levels)
    get_adjacent_values = get_adjacent_values_along_axis(num_inputs,levels)
    D = dimension
    
    @wp.kernel
    def laplacian_kernel(grid_points:wp.array2d(dtype = float),
                        current_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        alpha:float,
                        dimension:int,
                        levels:wp.vec3i,
                        new_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        ):
        
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        x_id += levels[0]
        y_id += levels[1]
        z_id += levels[2]
        
        node_ID = wp.vec(length = 3,dtype = int)
        node_ID[0] = x_id
        node_ID[1] = y_id
        node_ID[2] = z_id
        
        # Step 2. Get current indexed grid point and value
        current_point = wp.vec(length = 3,dtype = float)
        current_point[0] = grid_points[0,x_id]
        current_point[1] = grid_points[1,y_id]
        current_point[2] = grid_points[2,z_id]
        
        current_value = current_values[batch_id,x_id,y_id,z_id]
        laplace = wp.vec(length = wp.static(num_inputs),dtype = float)
        for axis in range(wp.static(D)):
            adj_points = get_adjacent_points_along_axis(grid_points,node_ID[axis],axis,levels[axis])
            # print(adj_points)
            adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
            
            laplace += second_order.central_difference(adj_points[1],current_point[axis],adj_points[0], adj_values[1],current_value,adj_values[0]) 
        
        new_values[batch_id,x_id,y_id,z_id] = alpha*laplace
        
    return laplacian_kernel



def create_grad_kernel(dimension,levels,num_outputs):
    
    # D = dimension
    # if num_outputs is not None:
    #     assert isinstance(num_outputs,int)
    #     assert num_outputs <= 3 , 'Number of output variable must be at most 3'
    #     assert num_outputs >= D ,'number of outputs for grad operator must be less than or equal to the specified dimension of the grid'
    # else:
    #     num_outputs = dimension
        
    
    get_adjacent_points_along_axis = get_adjacent_points(levels)
    get_adjacent_values = get_adjacent_values_along_axis(1,levels)
    D = dimension
    
    @wp.kernel
    def grad_kernel(grid_points:wp.array2d(dtype = float),
                        current_values:wp.array4d(dtype = wp.vec(length=1,dtype=float)),
                        alpha:float,
                        dimension:int,
                        levels:wp.vec3i,
                        new_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        ):
        
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        x_id += levels[0]
        y_id += levels[1]
        z_id += levels[2]
        
        node_ID = wp.vec3i(x_id,y_id,z_id)
        
        
        # Step 2. Get current indexed grid point and value
        current_point = wp.vec(length = 3,dtype = float)
        current_point[0] = grid_points[0,x_id]
        current_point[1] = grid_points[1,y_id]
        current_point[2] = grid_points[2,z_id]
        # wp.printf('Location %d %d %d   %f %f %f \n',x_id,y_id,z_id,current_point[0],current_point[1],current_point[2])
        current_value = current_values[batch_id,x_id,y_id,z_id]
        
    
        for axis in range(wp.static(D)):
            adj_points = get_adjacent_points_along_axis(grid_points,node_ID[axis],axis,levels[axis])
            # print(adj_points)
            adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
            new_values[batch_id,x_id,y_id,z_id][axis] = alpha*first_order.central_difference(adj_points[1],current_point[axis],adj_points[0], adj_values[1],current_value,adj_values[0])[0]
        
    return grad_kernel



def create_Divergence_kernel(dimension,levels,num_inputs):
    
        
    D = dimension
    get_adjacent_points_along_axis = get_adjacent_points(levels)
    get_adjacent_values = get_adjacent_values_along_axis(num_inputs,levels)
    
    @wp.kernel
    def divergence_kernel(grid_points:wp.array2d(dtype = float),
                        current_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        alpha:float,
                        dimension:int,
                        levels:wp.vec3i,
                        new_values:wp.array4d(dtype = wp.vec(length=1,dtype=float)),
                        ):
        
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        
        # Step 1. Shift to adjust for ghost cells
        x_id += levels[0]
        y_id += levels[1]
        z_id += levels[2]
        
        node_ID = wp.vec3i(x_id,y_id,z_id)
        
        
        # Step 2. Get current indexed grid point and value
        current_point = wp.vec(length = 3,dtype = float)
        current_point[0] = grid_points[0,x_id]
        current_point[1] = grid_points[1,y_id]
        current_point[2] = grid_points[2,z_id]
        # wp.printf('Location %d %d %d   %f %f %f \n',x_id,y_id,z_id,current_point[0],current_point[1],current_point[2])
        current_value = current_values[batch_id,x_id,y_id,z_id]
        
        
        # Saftey incase we pass in an array that isnt zeroed
        div_val = 0.
        for axis in range(wp.static(D)):
            adj_points = get_adjacent_points_along_axis(grid_points,node_ID[axis],axis,levels[axis])
            # print(adj_points)
            adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
            
            # We only need the dirivative of the varible in the axis direction only
            div_val += alpha*first_order.central_difference(adj_points[1],current_point[axis],adj_points[0], adj_values[1][axis],current_value[axis],adj_values[0][axis])
        
        new_values[batch_id,x_id,y_id,z_id][0] = div_val
        
    return divergence_kernel
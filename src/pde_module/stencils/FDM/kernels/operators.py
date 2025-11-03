from . import second_order
from . import first_order
from .neighbors import get_adjacent_values_along_axis,get_adjacent_points,get_adjacent_matrix_values_along_axis
import warp as wp


def create_Laplacian_kernel(dimension,num_inputs,levels):
    '''
    We need to ensure num_inputs == num_outputs
    '''
    # get_adjacent_points_along_axis = get_adjacent_points_along_axis_function(levels)
    
    get_adjacent_values = get_adjacent_values_along_axis(num_inputs,levels)
    D = dimension
    
    @wp.kernel
    def laplacian_kernel(
                        current_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        dx:float,
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
        
        
        current_value = current_values[batch_id,x_id,y_id,z_id]
        laplace = wp.vec(length = wp.static(num_inputs),dtype = float)
        for axis in range(wp.static(D)):
            # print(adj_points)
            adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
            laplace += second_order.central_difference(adj_values[1],current_value,adj_values[0],dx) 
        
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



def create_Divergence_kernel(dimension,levels,num_inputs,div_type):
        
    assert div_type in ['vector','tensor'], 'div_type is a string of either vector or tensor'
    
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



def create_tensor_divergence_kernel(dimension,levels,mat_shape,div_type):
        
    assert div_type in ['vector','tensor'], 'div_type is a string of either vector or tensor'
    
    
    num_outputs,C = mat_shape
    D = dimension
    
    assert C == D
    get_adjacent_points_along_axis = get_adjacent_points(levels)
    get_adjacent_values = get_adjacent_matrix_values_along_axis(mat_shape,levels)
    
    
    
    @wp.kernel
    def divergence_kernel(grid_points:wp.array2d(dtype = float),
                        current_values:wp.array4d(dtype = wp.mat(shape=mat_shape,dtype=float)),
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
        
        
        
        # Saftey incase we pass in an array that isnt zeroed
        # for output in range(wp.static(num_outputs)):
        div_val = wp.vec(length = num_outputs,dtype = float)
        for axis in range(wp.static(D)):
            adj_points = get_adjacent_points_along_axis(grid_points,node_ID[axis],axis,levels[axis])
            # print(adj_points)
            adj_values = get_adjacent_values(current_values,batch_id,x_id,y_id,z_id,axis)
            # wp.printf('row_1 %d %f %f\n',axis,adj_values[0,0],adj_values[0,1])
            # wp.printf('row_2 %d %f %f\n',axis,adj_values[1,0],adj_values[1,1])
            # wp.printf('adj %d %f %f\n',axis,adj_points[0],adj_points[1])
            div_val += alpha*first_order.central_difference(adj_points[1],current_point[axis],adj_points[0], adj_values[1],current_value[:,axis],adj_values[0])
            # wp.printf('%f %f\n',current_value[0,axis],current_value[1,axis])
            # wp.printf('%d %f %f\n',axis,div_val[0],div_val[1])
            # # We only need the dirivative of the varible in the axis direction only
            # div_val += alpha*first_order.central_difference(adj_points[1],current_point[axis],adj_points[0], adj_values[1][axis],current_value[axis],adj_values[0][axis])
        
        new_values[batch_id,x_id,y_id,z_id] = div_val
        
    return divergence_kernel



def create_vector_outer(len_a,len_b):

    @wp.func
    def outer_product(a:wp.vec(length = len_a,dtype = float),b:wp.vec(length = len_b,dtype = float)):
            
        mat = wp.mat(shape = (len_a,len_b),dtype = a.dtype)
        
        for i in range(len_a):
            mat[i] = a[i]*b
        return mat
    
    
    return outer_product


def create_outer_product_kernel(len_a,len_b):
    
    outer_product = create_vector_outer(len_a,len_b)
    
    @wp.kernel
    def outer_product_kernel(
                        vec_a:wp.array4d(dtype = wp.vec(length=len_a,dtype=float)),
                        vec_b:wp.array4d(dtype = wp.vec(length=len_b,dtype=float)),
                        scale:float,
                        new_values:wp.array4d(dtype = wp.mat(shape=(len_a,len_b),dtype=float)),
                        ):
        
        batch_id,x_id,y_id,z_id = wp.tid()
        
        new_values[batch_id,x_id,y_id,z_id] = scale*outer_product(vec_a[batch_id,x_id,y_id,z_id],vec_b[batch_id,x_id,y_id,z_id])
        
    return outer_product_kernel

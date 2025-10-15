import warp as wp
from typing import Any


def check_levels(levels):
    if isinstance(levels,(tuple,list)): # is wp.vector:
        return max(levels)
        
    elif hasattr(levels,'_wp_generic_type_str_'):
        if levels._wp_generic_type_str_ == 'vec_t':
            return max(levels)
        else:
            TypeError(f'Expected either tuple, warp.vector or int but got {type(levels)} with {levels._wp_generic_type_str_= } instead')
        
    elif isinstance(levels,int):
        return levels # I
    else:
        raise TypeError(f'Expected either tuple, warp.vector or int but got {type(levels)} instead')



def get_adjacent_points(levels):
    max_neighbors = check_levels(levels)
    

    @wp.func
    def _get_adjacent_points(grid_points:wp.array2d(dtype = float),node_ID:int,axis:int,levels:int):
        num_points = wp.static(max_neighbors*2)
        adj_points = wp.vec(length = num_points,dtype =float)
        
        starting_level = -levels
        # index_axis_vec = wp.vec3i(0,0,0)
        
        for i in range(levels*2):
            level = starting_level+i
            if level >= 0:
                level += 1
            adj_points[i] = grid_points[axis,node_ID+level]
            
            
            
            
        return adj_points
    
            
    return _get_adjacent_points



def get_adjacent_values_along_axis(num_inputs,levels):
    
    max_neighbors  = check_levels(levels)
    
    @wp.func
    def get_adjacent_values(current_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),batch_id:int,x_id:int,y_id:int,z_id:int,axis:int):
        num_points = wp.static(max_neighbors*2)
        inc_vec =wp.vec(length = 3,dtype = int)
        mat = wp.mat(shape=(num_points,num_inputs),dtype = float)
        
        starting_level = -max_neighbors
        for i in range(num_points):
            level = starting_level+i    
            if level >= 0:
                level += 1

            inc_vec[axis] = level
            
            for output in range(num_inputs):
                mat[i,output] = current_values[batch_id,x_id+inc_vec[0],y_id+inc_vec[1],z_id+inc_vec[2]][output]

        return mat
    
    return get_adjacent_values



def get_adjacent_matrix_values_along_axis(mat_shape,levels):
    
    max_neighbors  = check_levels(levels)
    num_inputs,num_rows = mat_shape
    @wp.func
    def get_adjacent_values(current_values:wp.array4d(dtype = wp.mat(shape=mat_shape,dtype=float)),batch_id:int,x_id:int,y_id:int,z_id:int,axis:int):
        '''
        I have a matrix at each point, i need to get the ith row and store it
        
        '''
        
        num_points = wp.static(max_neighbors*2)
        inc_vec =wp.vec(length = 3,dtype = int)
        mat = wp.mat(shape=(num_points,num_inputs),dtype = float)
        
        starting_level = -max_neighbors
        for i in range(num_points):
            level = starting_level+i    
            if level >= 0:
                level += 1

            inc_vec[axis] = level
            
            for output in range(num_inputs):
                mat[i,output] = current_values[batch_id,x_id+inc_vec[0],y_id+inc_vec[1],z_id+inc_vec[2]][output,axis]

        return mat
    
    return get_adjacent_values
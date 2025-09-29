import warp as wp
from typing import Any


def get_adjacent_values_along_axis(num_outputs,levels = 1):
    @wp.func
    def get_adjacent_values(current_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),batch_id:int,x_id:int,y_id:int,z_id:int,axis:int):
        num_points = wp.static(levels*2)
        inc_vec =wp.vec(length = 3,dtype = int)
        mat = wp.mat(shape=(num_points,num_outputs),dtype = float)
        
        starting_level = -levels
        for i in range(num_points):
            level = starting_level+i    
            if level >= 0:
                level += 1

            inc_vec[axis] = level
            
            for output in range(num_outputs):
                mat[i,output] = current_values[batch_id,x_id+inc_vec[0],y_id+inc_vec[1],z_id+inc_vec[2]][output]

        return mat
    
    return get_adjacent_values



def get_adjacent_points_along_axis_function(levels = 1):
    
    @wp.func
    def get_adjacent_points_along_axis(grid_points:wp.array3d(dtype = wp.vec(length=3,dtype=float)),
                                       x_id:int,
                                       y_id:int,
                                       z_id:int,
                                       axis:int):
        
        num_points = wp.static(levels*2)
        adj_points = wp.mat(shape = (num_points,3),dtype = float)
        
        starting_level = -levels
        index_axis_vec = wp.vec3i(0,0,0)
        
        for i in range(num_points):
            level = starting_level+i
            if level >= 0:
                level += 1
            index_axis_vec[axis] = level
            for j in range(3):
                adj_points[i,j] = grid_points[x_id+index_axis_vec[0],y_id+index_axis_vec[1],z_id +index_axis_vec[2]][j]
            
        return adj_points
    
    return get_adjacent_points_along_axis


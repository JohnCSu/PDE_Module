import warp as wp
from typing import Any





@wp.func
def dirichlet_ghost_cell_correction(boundary_value:float,adj_value:float,interior:bool):    
    if interior: # so interior point is +1
        return 2.*boundary_value - adj_value
    else:
        return 2.*boundary_value + adj_value


def create_boundary_kernel(num_outputs):
    @wp.kernel
    def boundary_kernel(grid_points:wp.array2d(dtype = float),
                        current_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
                        boundary_indices:wp.array2d(dtype=int),
                        boundary_type:wp.array(dtype=wp.vec(length=num_outputs,dtype=int)),
                        boundary_value:wp.array(dtype=wp.vec(length=num_outputs,dtype=float)),
                        interior_indices: wp.array(dtype = int),
                        interior_adjaceny:wp.array(dtype = wp.vec3i),
                        levels:wp.vec3i,
                        new_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float))):
        
        
        batch_id,i,output = wp.tid() # Loop through boundary_types
        
        
        x_id = boundary_indices[i,0]
        y_id = boundary_indices[i,1]
        z_id = boundary_indices[i,2]
            
        # Step 1. Shift to adjust fro stencil shape
        x_id += levels[0]
        y_id += levels[1]
        z_id += levels[2]
        
        nodeID = wp.vec3i(x_id,y_id,z_id)
        
        
        adj_index = interior_indices[i]
        adj_vec = interior_adjaceny[adj_index]
        if boundary_type[i][output] == 0: # Dirichlet
            # Dirichlet so just set the value
            new_values[batch_id,x_id,y_id,z_id][output] = boundary_value[i][output]
                
            #Update Ghost value
            for j in range(3):
                if adj_vec[j] != 0:                    
                    inc_vec = wp.vec3i(0,0,0)
                    inc_vec[j] = adj_vec[j]
                    ghostID = nodeID - inc_vec
                    adjID = nodeID + inc_vec
                    new_values[batch_id,ghostID[0],ghostID[1],ghostID[2]][output] = 2.*boundary_value[i][output] - current_values[batch_id,adjID[0],adjID[1],adjID[2]][output]

        elif boundary_type[i][output] == 1: # Von neumann
            
            #Update Ghost value
            for axis in range(3):
                if adj_vec[axis] != 0:                    
                    inc_vec = wp.vec3i()
                    
                    inc_vec[axis] = adj_vec[axis]
                    ghostID = nodeID - inc_vec
                    adjID = nodeID + inc_vec
                    
                    h = wp.abs(grid_points[axis,ghostID[axis]] - grid_points[axis,adjID[axis]]) 
                    new_values[batch_id,ghostID[0],ghostID[1],ghostID[2]][output] = boundary_value[i][output] - wp.sign(grid_points.dtype(adj_vec[axis]))*current_values[batch_id,adjID[0],adjID[1],adjID[2]][output]*h
            

    return boundary_kernel
    


@wp.func
def get_ghost_and_adj_ID(adj_vec:wp.vec3i,nodeID:wp.vec3i,axis:int):
    inc_vec = wp.vec3i(0,0,0)
    inc_vec[axis] = adj_vec[axis]
    ghostID = nodeID - inc_vec
    adjID = nodeID + inc_vec
    return ghostID,adjID
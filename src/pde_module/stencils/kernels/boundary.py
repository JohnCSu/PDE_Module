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
    def boundary_kernel(current_values:wp.array4d(dtype = wp.vec(length=num_outputs,dtype=float)),
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
        
        if boundary_type[i][output] == 0:
            # Dirichlet so just set the value
            new_values[batch_id,x_id,y_id,z_id][output] = boundary_value[i][output]
            # get the adj
            adj_index = interior_indices[i]
            adj_vec = interior_adjaceny[adj_index]
            
            for j in range(3):
                if adj_vec[j] != 0:                    
                    inc_vec = wp.vec3i(0,0,0)
                    inc_vec[j] = adj_vec[j]
                    ghostID = nodeID - inc_vec
                    adjID = nodeID + inc_vec
                    new_values[batch_id,ghostID[0],ghostID[1],ghostID[2]][output] = 2.*boundary_value[i][output] - new_values[batch_id,adjID[0],adjID[1],adjID[2]][output]

            
            
            

    return boundary_kernel
    
    
import warp as wp

def create_streaming_kernel(dimension,num_distributions):
    
    dist_vec = wp.vec(length = num_distributions,dtype = float)
    dist_mat = wp.mat(shape = (num_distributions,dimension),dtype = int)
    
    @wp.kernel
    def streaming_kernel(f_old:wp.array4d(dtype = dist_vec),discrete_velocities:dist_mat,f_new:wp.array4d(dtype = dist_vec)):
        batch_id,x_id,y_id,z_id = wp.tid()
        
        # nodeID = wp.vec(length = 2,dtype = int)(x_id,y_id)
        nodeID = wp.vec3i(x_id,y_id,z_id)
        
        for i in range(wp.static(len(discrete_velocities))):
            velocity_direction = discrete_velocities[i]
            
            
            pulled_idx = wp.vec3i(0,0,0)
            for j in range(dimension):
                pulled_idx[j] = nodeID[j] - velocity_direction[j] 
                if pulled_idx[j] < 0 or pulled_idx[j] >= f_old.shape[j]:
                    # We dont stream so just set pulled_idx to same as nodeID
                    pulled_idx = nodeID
                    break
                    
                    
            f_new[batch_id,x_id,y_id,z_id][i] = f_old[batch_id,pulled_idx[0],pulled_idx[1],pulled_idx[2]][i]
            
            
    return streaming_kernel
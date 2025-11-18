import warp as wp
from pde_module.grids.latticeModels import LatticeModel


def create_LBM_BC(dimension,lattice_model:LatticeModel):
    num_distributions = lattice_model.num_discrete_velocities
    dist_vec = wp.vec(length = num_distributions,dtype = float)
    dist_mat = wp.mat(shape = (num_distributions,dimension),dtype = float)
    num_dist_per_wall = lattice_model.into_wall_indices._shape_[0]
    
    
    into_wall_mat = wp.mat(shape = lattice_model.into_wall_indices._shape_,dtype = int)
    @wp.kernel
    def LBM_boundary_condition_kernel(f_old:wp.array4d(dtype = dist_vec),
                                      ownerNeighbor:wp.array2d(dtype=int),
                                      boundary_type:wp.array(dtype=wp.int8),
                                      boundary_value:wp.array(dtype=wp.vec(2,float)),
                                      wall_normal_IDs:wp.array(dtype=wp.int32),
                                      discrete_velocities:dist_mat,
                                            discrete_weights:dist_vec,
                                            reference_density:float,
                                            cs:float,
                                            opposite_indices:dist_vec,
                                            into_wall_indices:into_wall_mat,
                                            f_new:wp.array4d(dtype = dist_vec)):
        batch_id,face_id = wp.tid()
        
        cellID = ownerNeighbor[face_id,0]
        wall_normal_ID = wall_normal_IDs[face_id]
        into_wall_velocities = into_wall_indices[wall_normal_ID]
        
        if boundary_type[face_id] == 0: # Half-way BounceBack
            for i in wp.static(range(num_dist_per_wall)) : # number of reflecting distribution
                into_wall_dist_id = into_wall_velocities[i]
                refelction_id = opposite_indices[into_wall_dist_id]
                f_new[batch_id,cellID[0],cellID[1],cellID[2]][refelction_id] = f_old[batch_id,cellID[0],cellID[1],cellID[2]][into_wall_dist_id]
        
        elif boundary_type[face_id] == 1: # Moving Wall
            for i in wp.static(range(num_dist_per_wall)) : # number of reflecting distribution
                into_wall_dist_id = into_wall_velocities[i]
                refelction_id = opposite_indices[into_wall_dist_id]
                f_new[batch_id,cellID[0],cellID[1],cellID[2]][refelction_id] = f_old[batch_id,cellID[0],cellID[1],cellID[2]][into_wall_dist_id]  - 2.*reference_density*discrete_weights[into_wall_dist_id]/wp.pow(cs,2)*(wp.dot(discrete_velocities[into_wall_dist_id],boundary_value[face_id]))
                
        
        return LBM_boundary_condition_kernel
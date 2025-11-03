import warp as wp
from typing import Any


def create_BGK_kernel(dimension,num_distributions):
    
    
    dist_vec = wp.vec(length = num_distributions,dtype = float)
    dist_mat = wp.mat(shape = (num_distributions,dimension),dtype = float)
    
    
    @wp.func
    def f_eq(density:float,velocity:wp.vec(dimension,float),discrete_velocity:Any,discrete_weight:float):
        e_dot_u = wp.dot(velocity,discrete_velocity)
        return discrete_weight*density*(1. + 3.*e_dot_u + 4.5*e_dot_u**2. - 1.5*wp.dot(velocity,velocity))
    
    
    @wp.kernel
    def BGK_kernel(f_old:wp.array3d(dtype = dist_vec),
                   density_field:wp.array(ndim=dimension+1,dtype= wp.vec(1,float)),
                   velocity_field:wp.array(ndim=dimension+1,dtype= wp.vec(dimension,float)),
                   discrete_velocities:dist_mat,
                   discrete_weights:dist_vec,
                   relaxation_time:float,
                   f_new:wp.array3d(dtype = dist_vec)):
        
        
        batch_id,x_id,y_id = wp.tid()
        
        density = density_field[batch_id,x_id,y_id][0]
        velocity = velocity_field[batch_id,x_id,y_id]
        for i in range(wp.static(len(discrete_velocities))):
            f_new[batch_id,x_id,y_id][i] = f_old[batch_id,x_id,y_id][i] - 1./relaxation_time * (f_old[batch_id,x_id,y_id][i] - f_eq(density,velocity,discrete_velocities[i],discrete_weights[i]))
            


    return BGK_kernel
        
        
if __name__ == '__main__':
    create_BGK_kernel(2,9)
        
        wp.launch
    
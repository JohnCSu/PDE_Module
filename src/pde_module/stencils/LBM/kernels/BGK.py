import warp as wp



def create_BGK_kernel(dimension,num_distributions):
    
    
    @wp.func
    def f_eq(density,velocity,discrete_velocities,discrete_weights,i):
        e_dot_u = wp.dot(velocity,discrete_velocities)
        discrete_weights[i]*density*(1. + 3.*e_dot_u + 4.5*e_dot_u**2. - 1.5*wp.dot(velocity,velocity))
    
    
    @wp.kernel
    def BGK_kernel(f_old,density_field,velocity_field,discrete_velocities,discrete_weights,relaxation_time:float,f_new):
        batch_id,x_id,y_id = wp.tid()
        
        density = density_field[batch_id,x_id,y_id]
        velocity = velocity_field[batch_id,x_id,y_id]
        for i in len(discrete_velocities):
            f_new[batch_id,x_id,y_id][i] = f_old[batch_id,x_id,y_id][i] - 1./relaxation_time * (f_old[batch_id,x_id,y_id][i] - f_eq(density,velocity,discrete_velocities,discrete_weights))
            


    return BGK_kernel
        
        
def create_Streaming_kernel(dimension,num_distributions):
    
    def 
        
        
    
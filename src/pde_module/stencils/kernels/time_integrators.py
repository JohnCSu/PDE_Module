import warp as wp

def create_forward_euler_kernel(num_inputs):
    '''Most simple time integrator i.e. f(t+dt) = f(t) + dt*f'(t) where f'(t) is the final stencil'''
    @wp.kernel
    def forward_euler_kernel(
                        current_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        stencil_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        new_values:wp.array4d(dtype = wp.vec(length=num_inputs,dtype=float)),
                        dt:float):
        
        batch_id,x_id,y_id,z_id = wp.tid() # Lets only do internal grid points
        new_values[batch_id,x_id,y_id,z_id ] = current_values[batch_id,x_id,y_id,z_id ] + dt*stencil_values[batch_id,x_id,y_id,z_id ]
        
        
    return forward_euler_kernel

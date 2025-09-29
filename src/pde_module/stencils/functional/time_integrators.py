import warp as wp

def forward_euler(kernel,current_values,new_values,stencil_values,dt):
    '''Most simple time integrator i.e. f(t+dt) = f(t) + dt*f'(t) where f'(t) is the final stencil'''
    wp.launch(kernel,dim = current_values.shape,inputs = [current_values,stencil_values,new_values,dt])
    return new_values


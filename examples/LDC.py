import numpy as np
import warp as wp


from pde_module.stencils.module.operators import Laplacian,Grad
from pde_module.stencils.module.time_integrators import ForwardEuler
from pde_module.stencils.module.boundary import GridBoundary

wp.init()
wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from pde_module.grids import NodeGrid


if __name__ == '__main__':
    x,y = np.linspace(0,1,100),np.linspace(0,1,100)
    grid = NodeGrid(x,y)

    IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    
    initial_value =grid.initial_condition(IC)
    print(initial_value.shape)
    # initial_value += 1.
    levels = np.linspace(-0.15,1.05,100,endpoint=True)
    to_plot = grid.trim_ghost_values(initial_value.numpy())

    t = 0
    
    dx = x[1] - x[0]
    dt = float(dx**2/(4*0.1))
    grid.to_warp()
    
    
    p_field = grid.create_grid_with_ghost(1)
    u_field = grid.create_grid_with_ghost(3)
    

    u_boundary = GridBoundary(grid,3,1,dynamic_array_alloc= False)
    u_boundary.dirichlet_BC('-X',0.)
    u_boundary.dirichlet_BC('-Y',0.)
    u_boundary.dirichlet_BC('+X',0.)
    u_boundary.dirichlet_BC('+Y',1.,0)
    u_boundary.to_warp()
    
    u_diffusion = Laplacian(grid,3,1,dynamic_array_alloc=False)
    
    # p_grad = Grad(grid,1,dynamic_array_alloc= False)
    
    time_step = ForwardEuler(grid,1,dynamic_array_alloc= True)
    
        
    
    
    
    time_step.init_output_array(initial_value)
    laplcian_stencil.init_output_array(initial_value)
    boundary.init_output_array(initial_value)
    
    
    
    
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    input_values = initial_value
    print(input_values.numpy()[0,:,:,0,0])
    for i in range(4000):
        boundary_corrected_values = boundary(input_values)        
        # print(boundary_corrected_values.numpy()[0,:,:,0,0])
        laplace = laplcian_stencil(boundary_corrected_values,alpha =0.1)
        
        # print(laplace.numpy()[0,:,:,0,0])
        new_value = time_step(boundary_corrected_values,laplace,dt)
        
        input_values = new_value
        # print(input_values.numpy()[0,:,:,0,0])
        t += dt
        if t > 1.:
            break
        
    print(f't = {t:.3e} max value = {np.max(input_values.numpy().max()):.3E}, dt = {dt:.3E}')
    

    
    
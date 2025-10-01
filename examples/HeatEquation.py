import numpy as np
import warp as wp


from pde_module.stencils.FDM.module.operators import Laplacian,Grad,Divergence
from pde_module.stencils.FDM.module.time_integrators import ForwardEuler
from pde_module.stencils.FDM.module.boundary import GridBoundary

wp.init()
wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from pde_module.grids import NodeGrid


if __name__ == '__main__':
    n = 500
    x,y = np.linspace(0,1,n),np.linspace(0,1,n)
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
    

    boundary = GridBoundary(grid,1,dynamic_array_alloc= False)
    boundary.dirichlet_BC('ALL',0.)
    boundary.to_warp()
    
    laplcian_stencil = Laplacian(grid,1,dynamic_array_alloc=False)
    
    #Equivalent Hopefull
    grad_stencil = Grad(grid,dynamic_array_alloc= False)
    div_stencil = Divergence(grid,dynamic_array_alloc= False)
    
    
    time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    
        
    
    time_step.init_output_array(initial_value)
    laplcian_stencil.init_output_array(initial_value)
    boundary.init_output_array(initial_value)
    
    
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    input_values = initial_value
    print(input_values.numpy()[0,:,:,0,0])
    for i in range(1000):
        boundary_corrected_values = boundary(input_values)        
        laplace = laplcian_stencil(boundary_corrected_values,alpha =0.1)
        new_value = time_step(boundary_corrected_values,laplace,dt)
        input_values = new_value

        t += dt
        if t > 1.:
            break
        
    print(f't = {t:.3e} max value = {np.max(input_values.numpy().max()):.3E}, dt = {dt:.3E}')
    

    
    
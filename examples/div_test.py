import numpy as np
import warp as wp


from pde_module.stencils.FDM.module.operators import Laplacian,Grad,Divergence,RowWiseDivergence,OuterProduct
from pde_module.stencils.FDM.module.time_integrators import ForwardEuler
from pde_module.stencils.FDM.module.boundary import GridBoundary

wp.init()
wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from pde_module.grids import NodeGrid


if __name__ == '__main__':
    n = 3
    x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = NodeGrid(x,y)

    IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    
    initial_value =grid.initial_condition(IC)
    print(initial_value.shape)
    to_plot = grid.trim_ghost_values(initial_value)

    t = 0
    
    dx = x[1] - x[0]
    dt = float(dx**2/(4*0.1))
    grid.to_warp()
    
    boundary = GridBoundary(grid,1,dynamic_array_alloc= False)
    boundary.dirichlet_BC('ALL',0.)
    
    laplcian_stencil = Laplacian(grid,1,dynamic_array_alloc=False)
    
    #Equivalent Hopefull
    
    div_stencil = Divergence(grid,dynamic_array_alloc= False)
    
    
    
    
    # mat_field = grid.create_grid_with_ghost((1,2))
    # mat_field.fill_(1.)
    
    u = grid.create_grid_with_ghost(2)
    u.fill_(1.)
    
    outer = OuterProduct(grid,2,2,dynamic_array_alloc=False)
    
    
    u_div_row = RowWiseDivergence(grid,(2,2),dynamic_array_alloc= False)
    
    time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    input_values = initial_value
    # print(input_values.numpy()[0,:,:,0,0])
    print(grid.stencil_points)
    for i in range(1):
        print(u.numpy()[0,:,:,0,0])
        u_out = outer(u,u)
        c = u_out.numpy()[0,:,:,0,:,:]
        
        print(c[:,:,0,0])
        d = np.sum(c,axis = -1)
        print(np.sum(d,axis=-1))
        
        u_div = u_div_row(u_out)
        print(u_div.numpy()[0,:,:,0,0])
        t += dt
        if t > 1.:
            break

    
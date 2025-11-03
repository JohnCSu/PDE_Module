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
from pde_module.grids.grid import Grid


if __name__ == '__main__':
    n = 31
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/n, nx = n, ny = n,levels = (1,1))
    
    

    IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    
    
    space = grid.create_grid_point_field(field_type='node',batch_size=2)
    print(space.shape)
    print(wp.array(space,dtype= wp.vec3).shape)
    initial_value =grid.initial_condition('node',IC)
    
    exit()
    
    # to_plot = grid.trim_ghost_values(initial_value)

    t = 0
    
    dx = grid.dx
    dt = float(dx**2/(4*0.1))
    grid.to_warp()
    
    boundary = GridBoundary(grid,1,dynamic_array_alloc= False)
    boundary.dirichlet_BC('ALL',0.)
    
    laplcian_stencil = Laplacian(grid,1,dynamic_array_alloc=False)
    
    #Equivalent Hopefull
    grad_stencil = Grad(grid,dynamic_array_alloc= False)
    div_stencil = Divergence(grid,dynamic_array_alloc= False)
    
    time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    input_values = initial_value
    print(input_values.numpy()[0,:,:,0,0])
    
    # for i in range(0):
    #     boundary_corrected_values = boundary(input_values)
    #     # print(boundary_corrected_values.numpy()[0,:,:,0,0])        
    #     laplace = laplcian_stencil(boundary_corrected_values,scale =0.1)
    #     new_value = time_step(boundary_corrected_values,laplace,dt)
    #     input_values = new_value

    #     t += dt
    #     if t > 1.:
    #         break
        
    # print(f't = {t:.3e} max value = {np.max(input_values.numpy().max()):.3E}, dt = {dt:.3E}')
    

    # to_plot = grid.trim_ghost_values(input_values)
    # u = to_plot[0,:,:,0]
    # meshgrid = grid.plt_meshgrid
    # meshgrid = [m.T for m in meshgrid]
    # # print(f'max u {np.max(u):.3E}')
    
    # # plt.quiver(*meshgrid[::-1],u.T,v.T)
    # # plt.show()
    
    # plt.contourf(*meshgrid[::-1],u.T,cmap ='jet',levels = 100)
    # plt.colorbar()
    # plt.show()
    
    
    
    # u_05 = to_plot[0,n//2,:,0]
    # plt.plot(y,u_05)
    # plt.show()
    
    # g = Grad(grid,dynamic_array_alloc=False)
    
    # grad_u = g(input_values)
    # grad_u = grid.trim_ghost_values(grad_u)
    
    
    
    # # At x = 0.5 we want df/dx, so y
    # gx_05 = grad_u[0,:,n//2,0]
    # plt.plot(y,gx_05,label = 'FD')
    
    # plt.plot(y,np.sin(np.pi*0.5)*np.pi*np.cos(np.pi*y),label = 'analytic')
    # plt.legend()
    # plt.show()
    
    
    
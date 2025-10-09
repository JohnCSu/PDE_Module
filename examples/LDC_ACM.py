import numpy as np
import warp as wp


from pde_module.stencils.FDM.module.operators import Laplacian,Grad,Divergence
from pde_module.stencils.FDM.module.time_integrators import ForwardEuler
from pde_module.stencils.FDM.module.boundary import GridBoundary
from pde_module.stencils.FDM.module.map import ElementWiseMap
wp.init()
# wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from pde_module.grids import NodeGrid

'''
LID DRIVEN CAVITY

This examples shows solving 2D Navier Stokes in the classic LDC example using the Artificial Compressibility Method for explicit CFD. 

Domain: 101 x 101 grid nodes
Re: 100 (All other constant are 1)

For this example no convection term.

Equations:

du/dt = d^2u/dx^2 - dp/dx
dv/dt = d^2v/dx^2 - dp/dx

dp/dt +beta^2 * div(u) = 0

we add some rhie chow correction to suppress checker boarding in the form:

div(u) = div(u*) - laplace(p)

where u* is the divergence of u at the current time step with no preessure correction

Constants to tune:
dt - time step 
beta - ACM term, higher means faster convergence but instability (akin to increasing time stepping)
CFL_LIMIT - constant to ensure CFL is not violated set to 0.1
'''


if __name__ == '__main__':
    n = 101
    x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = NodeGrid(x,y)

    t = 0
    
    dx = x[1] - x[0]
    Re = 100.
    viscosity = 1/Re
    beta = 0.5
    CFL_LIMIT = 0.1
    
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+beta))
    print(f'{dt=:.3E} {beta=:.3E} {viscosity=:.3E}')
    # dt = 1.e-3
    grid.to_warp()
    
    u_field = grid.create_grid_with_ghost(2)
    u_boundary = GridBoundary(grid,2,dynamic_array_alloc= False)
    u_boundary.dirichlet_BC('ALL',0.)
    u_boundary.dirichlet_BC('+Y',1.,0)
    
    u_diffusion = Laplacian(grid,2,dynamic_array_alloc=False)
    u_time_step = ForwardEuler(grid,2,dynamic_array_alloc= False)
    u_div = Divergence(grid)
    
    
    p_field = grid.create_grid_with_ghost(1)
    p_boundary = GridBoundary(grid,1,dynamic_array_alloc=False)
    p_boundary.vonNeumann_BC('ALL',0.)
    p_boundary.dirichlet_BC(0,0.)
    
    p_grad = Grad(grid,dynamic_array_alloc= False)
    p_time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    p_diffusion = Laplacian(grid,1,dynamic_array_alloc= False)
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    u = u_field
    p = p_field
    # print(input_values.numpy()[0,:,:,0,0])
    
    
    func = lambda x,y: x+y
    p_sum = ElementWiseMap(func,grid,dynamic_array_alloc=False)
    u_sum = ElementWiseMap(func,grid,dynamic_array_alloc=False)
    
    
    for i in range(10000):
        # Apply BC to u and p fields
        
        u = u_boundary(u)
        p = p_boundary(p)
                
        #Calculate u laplace and div
        u_diff = u_diffusion(u,scale = viscosity)
        #p_grad
        dp = p_grad(p,scale = -1.)
        div_u = u_div(u,scale = -1.)
        p_diff = p_diffusion(p,scale = dt)
        
        u_F = u_sum(dp,u_diff)
        p_F = p_sum(div_u,p_diff)
        
        u = u_time_step(u,u_F,dt)
        p = p_time_step(p,p_F,dt*beta**2.)
        
        t += dt
        
    print(f't = {t:.3e} max value = {np.max(u.numpy().max()):.3E}, dt = {dt:.3E}')
    

    to_plot = grid.trim_ghost_values(u)
    
    p_plot = grid.trim_ghost_values(p)
    u_plot = to_plot.squeeze()
    
    u = to_plot[0,:,:,0]
    v = to_plot[0,:,:,1]
    u_mag = np.sqrt(u**2+v**2)
    p = p_plot.squeeze()
    meshgrid = grid.plt_meshgrid
    meshgrid = [m.T for m in meshgrid]
    # print(f'max u {np.max(u):.3E}')
    
    plt.quiver(*meshgrid[::-1],u.T,v.T)
    plt.show()
    
    plt.contourf(*meshgrid[::-1],u_mag.T,cmap ='jet',levels = 100)
    plt.show()
    
    
    #Plot centerline velocities
    
    x_05 = meshgrid[0][:,n//2]
    v_05 = v[:,n//2]
    
    plt.plot(x_05,v_05)
    plt.show()
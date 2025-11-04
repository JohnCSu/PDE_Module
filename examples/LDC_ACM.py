import numpy as np
import warp as wp


from pde_module.stencils.FDM.module.operators import Laplacian,Grad,Divergence,OuterProduct,RowWiseDivergence
from pde_module.stencils.FDM.module.time_integrators import ForwardEuler
from pde_module.stencils.FDM.module.boundary import GridBoundary
from pde_module.stencils.FDM.module.map import ElementWiseMap
wp.init()
# wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from pde_module.grids import Grid

'''
LID DRIVEN CAVITY

This examples shows solving 2D Navier Stokes in the classic LDC example using the Artificial Compressibility Method for explicit CFD. 

Domain: 101 x 101 grid nodes
Re: 100 (All other constant are 1)

For this example no convection term.

Equations:

du/dt = d^2u/dx^2 - dp/dx
dv/dt = d^2v/dx^2 - dp/dx

dp/dt +beta * div(u) = 0


Constants to tune:
dt - time step 
beta - ACM term, higher means faster convergence but instability (akin to increasing time stepping)
CFL_LIMIT - constant to ensure CFL is not violated set to 0.5
'''


if __name__ == '__main__':
    n = 101 # Use Odd number of points!
    dx = 1/(n-1)
    grid = Grid('node',dx=dx,nx = n,ny = n,levels = 1)

    t = 0
    
    dx = grid.dx
    Re = 100.
    viscosity = 1/Re
    beta = 0.7
    CFL_LIMIT = 0.5
    
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    print(f'{dt=:.3E} {beta=:.3E} {viscosity=:.3E}')
    # dt = 1.e-3
    # grid.to_warp()
    
    u_field = grid.create_nodal_field(2)
    u_boundary = GridBoundary(grid,2,dynamic_array_alloc= False)
    u_boundary.dirichlet_BC('ALL',0.)
    u_boundary.dirichlet_BC('+Y',1.,0)
    
    u_diffusion = Laplacian(grid,2,dynamic_array_alloc=False)
    
    u_outer = OuterProduct(grid,2,2,dynamic_array_alloc= False)
    u_convec = RowWiseDivergence(grid,(2,2),dynamic_array_alloc=False)
    
    u_time_step = ForwardEuler(grid,2,dynamic_array_alloc= False)
    u_div = Divergence(grid)
    
    p_field = grid.create_nodal_field(1)
    p_boundary = GridBoundary(grid,1,dynamic_array_alloc=False)
    p_boundary.vonNeumann_BC('ALL',0.)
    
    
    p_grad = Grad(grid,dynamic_array_alloc= False)
    p_time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    p_diffusion = Laplacian(grid,1,dynamic_array_alloc= False)
    # with wp.ScopedCapture(device="cuda") as iteration_loop:
    np.set_printoptions(precision=2)
    u = u_field
    p = p_field
    # print(input_values.numpy()[0,:,:,0,0])
    
    
    # func = lambda x,y: x+y
    p_sum = ElementWiseMap(lambda x,y: x+y,grid,dynamic_array_alloc=False)
    # u_sum = ElementWiseMap(lambda x,y: x+y,grid,dynamic_array_alloc=False)
    u_sum = ElementWiseMap(lambda x,y,z: x+y+z,grid,dynamic_array_alloc=False)
    
    
    for i in range(10001):
        # Apply BC to u and p fields
        
        u = u_boundary(u)
        p = p_boundary(p)
                
        #Calculate u laplace and div
        u_diff = u_diffusion(u,scale = viscosity)
        # print(u_diff.numpy())
        u_out = u_outer(u,u)
        u_convection = u_convec(u_out,scale = -1.)
        dp = p_grad(p,scale = -1.)
        u_F = u_sum(dp,u_diff,u_convection)
        
        # For ACM Constraint
        div_u = u_div(u,scale = -1.)
        # p_diff = p_diffusion(p,scale = dt/2)
        p_F = div_u
        # p_F = p_sum(div_u,p_diff)
        p = p_time_step(p,p_F,dt*beta)
        u = u_time_step(u,u_F,dt)
        
        t += dt
        if i % 5000 == 0 :
            
            incompr = np.abs(grid.trim_ghost_values(div_u))
            
            print(f'Iteration {i}, t = {t} incomp: max {incompr.max()}, avg {incompr.mean()}')
    print(f't = {t:.3e} max value = {np.max(u.numpy().max()):.3E}, dt = {dt:.3E}')
    
    # exit()
    to_plot = grid.trim_ghost_values(u)
    
    p_plot = grid.trim_ghost_values(p)
    u_plot = to_plot.squeeze()
    
    u = u_plot[:,:,0]
    v = u_plot[:,:,1]
    u_mag = np.sqrt(u**2+v**2)
    p = p_plot.squeeze()
    meshgrid = grid.meshgrid('node',False,True)
    meshgrid = [m for m in meshgrid]
    # print(f'max u {np.max(u):.3E}')
    
    plt.quiver(*meshgrid[::-1],u.T,v.T)
    plt.show()
    
    plt.contourf(*meshgrid[::-1],u.T,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()
    
    plt.contourf(*meshgrid[::-1],v.T,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()
    
    plt.contourf(*meshgrid[::-1],u_mag.T,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()
    
    #Plot centerline velocities
    import pandas as pd
    v_benchmark = pd.read_csv(r'examples\v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples\u_velocity_results.txt',sep= '\t')
    
    
    x_05 = meshgrid[0][:,n//2]
    v_05 = v[:,n//2]
    
    
    print(f"CFD max {v_05.max()}, Benchmark Max :{v_benchmark['100'].max()}")
    plt.plot(v_benchmark['%x'],v_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(x_05,v_05)
    plt.show()
    
    y_05 = meshgrid[1][n//2,:]
    u_05 = u[n//2,:]
    
    
    print(f"CFD max {u_05.max()}, Benchmark Max :{u_benchmark['100'].max()}")
    plt.plot(u_benchmark['%y'],u_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(y_05,u_05)
    plt.show()
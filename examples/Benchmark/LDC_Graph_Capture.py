
'''
LID DRIVEN CAVITY

This examples shows solving 2D Navier Stokes in the classic LDC example using the Artificial Compressibility Method for explicit CFD. 

Domain: 101 x 101 grid nodes
Re: 100 (All other constant are 1)

Equations:

du/dt grad_u * u =  d^2u/dx^2 + d^2u/dy^2 - dp/dx

dp/dt +beta * div(u) = 0


Constants to tune:
dt - time step. Simple Euler Step is used
beta =0.3  ACM term, higher means faster convergence but instability (akin to increasing time stepping)
CFL_LIMIT = 0.5 - constant to ensure CFL is not violated set to 0.5

'''


import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.experimental.grid import Grid
from pde_module.experimental.FDM.laplacian import Laplacian
from pde_module.experimental.time_integrators import ForwardEuler
from pde_module.experimental.FDM.gridBoundary import GridBoundary
from pde_module.experimental.FDM.grad import Grad
from pde_module.experimental.FDM.divergence import Divergence

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 101
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params
    viscosity = 1/100
    beta = 0.3
    CFL_LIMIT = 0.5
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    
    # Define Modules
    
    u = grid.create_node_field(2)
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.dirichlet_BC('ALL',0.)
    u_BC.dirichlet_BC('+Y',1.,0)
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    
    u_div = Divergence('vector',u.shape,dx,ghost_cells)
    u_step = ForwardEuler(u.dtype)
    
    p = grid.create_node_field(1)
    p_BC = GridBoundary(p,dx,ghost_cells)
    p_BC.vonNeumann_BC('ALL',0.)
    p_grad = Grad(1,p.shape,dx,ghost_cells)
    p_step = ForwardEuler(p.dtype)
    t= 0

    # Inital Pass + Setup Of kernels
    wp.set_mempool_release_threshold("cuda:0", 0.25)
    u_fix = u_BC(u)
    u_diff = u_lapl(u_fix,viscosity)
    du = u_grad(u_fix)
    u_incomp = u_div(u_fix,-beta)
    
    u_conv = du*u
    
    p_fix = p_BC(p)
    dp = p_grad(p_fix,-1.)
    u_F = dp - u_conv + u_diff
    u_next = u_step(u_fix,u_F,dt)
    p_next = p_step(p_fix,u_incomp,dt)
    u,p = u_next,p_next
    t+= dt
    
    with wp.ScopedCapture() as capture:   
        #S1
        u_fix = u_BC(u)
        u_diff = u_lapl(u_fix,viscosity)
        du = u_grad(u_fix)
        u_incomp = u_div(u_fix,-beta)
        u_conv = du*u
        #S2
        p_fix = p_BC(p)
        dp = p_grad(p_fix,-1.)
        #S3
        u_F = dp - u_conv + u_diff
        u_next = u_step(u_fix,u_F,dt)
        #S4
        p_next = p_step(p_fix,u_incomp,dt)
        #Sync
        u,p = u_next,p_next
        
    with wp.ScopedTimer("LDC_Graph_Capture"):
        for i in range(1,10001):
            wp.capture_launch(capture.graph)
            t+= dt
            if i % 2500 == 0:
                print(f't = {t:.3E},iter = {i}')
            
    
    # Trim Values
    meshgrid,us = grid.get_plotting_for('node',u)
    u_plot = us[:,:,0]
    v_plot = us[:,:,1]
    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    plt.contourf(*meshgrid,u_mag,cmap='jet',levels = np.linspace(0,1,20))
    plt.colorbar()
    plt.show()
    
    import pandas as pd
    v_benchmark = pd.read_csv(r'examples\v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples\u_velocity_results.txt',sep= '\t')
    
    
    x_05 = meshgrid[0][:,n//2]
    v_05 = v_plot[:,n//2]
    
    
    print(f"CFD max {v_05.max()}, Benchmark Max :{v_benchmark['100'].max()}")
    plt.plot(v_benchmark['%x'],v_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(x_05,v_05)
    plt.show()
    
    y_05 = meshgrid[1][n//2,:]
    u_05 = u_plot[n//2,:]
    
    
    print(f"CFD max {u_05.max()}, Benchmark Max :{u_benchmark['100'].max()}")
    plt.plot(u_benchmark['%y'],u_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(y_05,u_05)
    plt.show()
    # plot
    # def plot_2D_field(field,grid):
        
'''
2D Heat Equation Example

A simple PDE where we solve the heat equation:

du/dt = alpha*laplace(u)

Over a 2D 1x1 Grid

For Initial Condition:
u(x,y,0) = sin(pi*x)*sin(pi*y)

With Boundary Condition at the grid perimeter:
u(x_b,y_b,t) = 0

This has an Analytic solution:

u(x,y,t) = exp(-2.*alpha*pi^2*t)sin(pi*x)*sin(pi*y)


'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM.laplacian import Laplacian
from pde_module.time_step.forwardEuler import ForwardEuler
from pde_module.FDM.gridBoundary import GridBoundary

wp.init()


if __name__ == '__main__':
    #Geometry
    n = 101
    L = 1
    dx = L/(n-1)
    
    #Params
    alpha = 0.1
    dt = float(dx**2/(4*alpha))
    ghost_cells = 1
    
    
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    u =grid.initial_condition('node',IC)
    
    # Define Modules
    
    BC = GridBoundary(u,dx,1)
    BC.dirichlet_BC('ALL',0.)
    
    euler_step = ForwardEuler(u.dtype)
    lapl = Laplacian(1,dx,ghost_cells)

    lapl.setup(u)
    BC.setup(u)
    
    t= 0
    for i in range(1000):
        u2 = BC(u)
        stencil =lapl(u2,alpha)
        u_next = euler_step(u2,stencil,dt)
        u = u_next
        t+= dt
        if i % 100 == 0:
            print(f'Max u = {u.numpy().max()} at t = {t},iter = {i}')
    
meshgrid,us = grid.get_plotting_for('node',u)
plt.contourf(*meshgrid,us,cmap ='jet',levels = 100)
plt.colorbar()
plt.show()

import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.experimental.grid import Grid
from pde_module.experimental.FDM.laplacian import Laplacian
from pde_module.experimental.time_integrators import ForwardEuler
from pde_module.experimental.FDM.gridBoundary import GridBoundary

from pde_module.experimental.FDM.immersedBoundary import ImmersedBoundary
from pde_module.experimental.FDM.grad import Grad
from pde_module.experimental.FDM.divergence import Divergence

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 501
    L = 6
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = dx,num_points=(3*n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params
    viscosity = 1/100
    beta = 0.3
    CFL_LIMIT = 1.
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    print(dt)
    R = 1.
    # cyl = lambda x,y,z: (x-L/2)**2 + (y-L/2)**2 <= R**2
    cyl = lambda x,y,z : (x-L/2)**2 + (y-L/2)**2 <= R**2
    meshgrid = grid.create_meshgrid('node')
    # Define Modules
    
    u = grid.create_node_field(2)
    u_BC = GridBoundary(u,dx,ghost_cells)
    # u_BC.dirichlet_BC('ALL',0.)
    u_BC.dirichlet_BC('-X',1.,0)
    u_BC.impermeable('+Y')
    u_BC.impermeable('-Y')
    u_BC.vonNeumann_BC('+X',0.)
    
    
    u_cyl = ImmersedBoundary(u,dx,ghost_cells)
    u_cyl.from_bool_func(cyl,meshgrid)
    u_cyl.finalize()
    u_cyl.dirichlet_BC('ALL',0.)
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    
    u_div = Divergence('vector',u.shape,dx,ghost_cells)
    u_step = ForwardEuler(u.dtype)
    
    p = grid.create_node_field(1)
    p_BC = GridBoundary(p,dx,ghost_cells)
    p_BC.vonNeumann_BC('ALL',0.)
    p_BC.dirichlet_BC('+X',0.)
    
    p_cyl = ImmersedBoundary(p,dx,ghost_cells)
    p_cyl.from_bool_func(cyl,meshgrid)
    p_cyl.finalize()
    p_cyl.vonNeumann_BC('ALL',0.)
    
    p_grad = Grad(1,p.shape,dx,ghost_cells)
    p_step = ForwardEuler(p.dtype)
    
    
    t= 0
    '''
    Equations:

    du/dt = d^2u/dx^2 - dp/dx
    dv/dt = d^2v/dx^2 - dp/dx

    dp/dt +beta * div(u) = 0
    '''
    # Inital Pass + Setup Of kernels
    # wp.set_mempool_release_threshold("cuda:0", 0.25)
    # u_ibm = u_cyl.setup(u)
    # u_fix = u_BC.setup(u)
    for i in range(0,10001 ):
        
        # U field Calcs
        u_ibm = u_cyl(u)
        u_fix = u_BC(u_ibm)
        u_diff = u_lapl(u_fix,viscosity)
        du = u_grad(u_fix)
        u_incomp = u_div(u_fix,-beta)
        u_conv = du*u
        
        #Pressure Field Calcs
        p_ibm = p_cyl(p)
        p_fix = p_BC(p_ibm)
        
        dp = p_grad(p_fix,-1.)
        u_F = dp - u_conv + u_diff
        u_next = u_step(u_fix,u_F,dt)
        p_next = p_step(p_fix,u_incomp,dt)
        u,p = u_next,p_next
        t+= dt
        
        if i % 1000 == 0 and i > 360:
            print(f'Max u = {u.numpy().max()} at t = {t},iter = {i}')
    
    
    
    
    
    # with wp.ScopedCapture() as capture:   
    #     u_ibm = u_cyl(u)
    #     u_fix = u_BC(u_ibm)
    #     u_diff = u_lapl(u_fix,viscosity)
    #     du = u_grad(u_fix)
    #     u_incomp = u_div(u_fix,-beta)
        
    #     u_conv = du*u
        
        
    #     p_ibm = p_cyl(p)
    #     p_fix = p_BC(p_ibm)
        
    #     dp = p_grad(p_fix,-1.)
    #     u_F = dp - u_conv + u_diff
    #     u_next = u_step(u_fix,u_F,dt)
    #     p_next = p_step(p_fix,u_incomp,dt)
    #     u,p = u_next,p_next
        
        
    # with wp.ScopedTimer("LDC"):
    #     for i in range(1,2):
    #         wp.capture_launch(capture.graph)
    #         if i % 1000 == 0:
    #             print(f'Max u = {u.numpy().max()} at t = {t},iter = {i}')
            
    
    # Trim Values
    # wp.synchronize()
    meshgrid,us = grid.get_plotting_for('node',u)
    u_plot = us[:,:,0]
    v_plot = us[:,:,1]
    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    plt.contourf(*meshgrid,u_mag,cmap='jet',levels = 15)
    plt.colorbar()
    
    solid_meshgrid = grid.create_meshgrid('node')
    X,Y = solid_meshgrid[:2]
    idx = tuple(u_cyl.solid_boundary.T)
    x,y = X[idx],Y[idx]
    plt.scatter(x,y,s = 4,label = 'Boundary',color = 'k')
    plt.gca().set_aspect('equal')
    
    
    plt.show()
    
        

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
from pde_module.experimental.FDM.elementWiseOps import scalarVectorMult,OuterProduct

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 51
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    U = 1.
    D = 1.
    R = D/2
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = dx,num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params
    viscosity = 1/100
    density = 1.
    M = 0.3
    cs = U/M
    c2 = cs**2
    
    # beta = 0.3
    beta = 1/(U**2/c2)
    CFL_LIMIT = 0.3
    
    d_vis = CFL_LIMIT*float(dx**2/(viscosity))
    d_conv = CFL_LIMIT*dx/c2
    d_acoustic = CFL_LIMIT*dx/(cs + U)
    
    # print(beta)
    dt = min(d_vis,d_conv,d_acoustic)
    # dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    
    print(dt)
    
    meshgrid = grid.create_meshgrid('node')
    # Define Modules
    
    u = grid.create_node_field(2)
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.dirichlet_BC('ALL',0.)
    u_BC.dirichlet_BC('+Y',1.,0)
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    u_step = ForwardEuler(u.dtype)
    
    
    u_outer = OuterProduct(2,2)
    u_conv_div = Divergence((2,2),u.shape,dx,ghost_cells)
    
    

    rho = grid.create_node_field(1)
    rho.fill_(1.)
    # u.fill_(1.)
    # print(rho.numpy().squeeze())
    rho_BC = GridBoundary(rho,dx,ghost_cells)
    rho_BC.vonNeumann_BC('ALL',0.)
    # print(rho_BC.boundary_type)
    # print(u_BC.boundary_type)
    # rho_BC.dirichlet_BC('ALL',density)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)
    rho_step = ForwardEuler(rho.dtype)
    
    np.set_printoptions(precision=3,suppress= False)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)

    rho_step = ForwardEuler(rho.dtype)
    
    momentum_div = Divergence(2,u.shape,dx,ghost_cells)
    momentum = scalarVectorMult(2)

    
    t= 0
    '''
    Equations:

    du/dt = d^2u/dx^2 - dp/dx
    dv/dt = d^2v/dx^2 - dp/dx

    dp/dt +beta * div(u) = 0
    '''
    # Inital Pass + Setup Of kernels
    wp.set_mempool_release_threshold("cuda:0", 0.50)
    # u_ibm = u_cyl.setup(u)
    # u_fix = u_BC.setup(u)
    
    for i in range(0,20001 ):    
        # Boundary
        
        u_fix = u_BC(u)
        rho_fix = rho_BC(rho)

        # Momentum
        m = momentum(rho_fix,u_fix)
        m_div = momentum_div(m,-1.)
        
        #Convec + diff
        u_diff = u_lapl(u_fix,viscosity)        
        u_2 = u_outer(m,u_fix)
        u_conv = u_conv_div(u_2)
        
        #Pressure Gradient
        p = cs**2*(rho.view(float) - density).view(rho.dtype)
        dp = p_grad(p,-1.)
        # Sum
        u_F = dp - u_conv + u_diff
        m_next = u_step(m,u_F,dt)
        
        # print(rho_fix.numpy().squeeze())
        # print(p.numpy().squeeze())
        # print(u_fix.numpy().squeeze()[:,:,0])
        
        
        rho_next = rho_step(rho_fix,m_div,dt)
        
        u_next = m_next/rho_next.view(float)
        
        u,rho = u_next,rho_next
        
        t+= dt
        
        if i % 100 == 0:
            print(f'Max p = {p.numpy().max()}, u = {u_next.numpy().max()} at t = {t},iter = {i}')
    

    # Trim Values
    meshgrid,us = grid.get_plotting_for('node',u)
    meshgrid,p = grid.get_plotting_for('node',p)
    u_plot = us[:,:,0]
    v_plot = us[:,:,1]
    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    plt.contourf(*meshgrid,u_mag,cmap='jet',levels = np.linspace(0,1,20))
    plt.colorbar()
    plt.show()
    
    plt.contourf(*meshgrid,p,cmap='jet',levels = 100)
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
        
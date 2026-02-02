
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
from pde_module.experimental.Stencil.mapWise import MapWise
from warp.types import vector,matrix,types_equal

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 201
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
    M = 0.2
    cs = U/M
    c2 = cs**2
    
    # beta = 0.3
    beta = 1/(U**2/c2)
    CFL_LIMIT = 0.75
    
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

    # wp.vec3f()
    
    @wp.func
    def get_p_op(rho:vector(1,float),c2:float,density:float):
        rho[0] = c2*(rho[0] - density)
        return rho
    
    get_p = MapWise(get_p_op)
    
    get_u_from_m = MapWise(lambda m,rho: m/rho[0])
    t= 0
    
    def f(t,u,rho):
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
        # p = cs**2*(rho.view(float) - density).view(rho.dtype)
        p = get_p(rho_fix,c2,density)
        # print(p.numpy().squeeze())
        dp = p_grad(p,-1.)
        # Sum
        u_F = dp - u_conv + u_diff
        m_next = u_step(m,u_F,dt)    
        rho_next = rho_step(rho_fix,m_div,dt)
        u_next = get_u_from_m(m_next,rho_next)
        u,rho = u_next,rho_next
        return u,rho
    
    # print(u.shape,p.shape)
    u,rho = f(t,u,rho)
    t+=dt
    with wp.ScopedCapture() as capture:
        u,rho = f(t,u,rho)

    # Create the animation
    
    meshgrid = grid.create_meshgrid('node')[:2]
    X,Y = [m.squeeze() for m in meshgrid]
    fig, ax = plt.subplots()
    im = ax.imshow(u.numpy().squeeze()[:,:,0].T,origin='lower', animated=True, cmap='jet',vmin = 0.,vmax = 1.)
    # im = ax.contourf(X,Y,,cmap= 'jet')
    fig.colorbar(im, ax=ax, label='U mag')
    
    step_per_frame = 250
    TIME = 0
    def render(frame,step_per_frame,dt):
        
        for i in range(step_per_frame):
            wp.capture_launch(capture.graph)
        
        t = frame*step_per_frame*dt
        new_u = u.numpy().squeeze()
        u_mag = (new_u[:,:,0]**2 + new_u[:,:,1]**2)**0.5
        ax.set_title(f'Step {frame*step_per_frame} Time = {t:.3f}')
        # step += 1
        im.set_data(u_mag.T)
        return [im]
        
    
    ani = FuncAnimation(fig,render , frames= 500, interval=1, repeat=False,fargs = [step_per_frame,dt])

    plt.show()
   
    meshgrid,us = grid.get_plotting_for('node',u)
    meshgrid,p = grid.get_plotting_for('node',rho)
    u_plot = us[:,:,0]
    v_plot = us[:,:,1]
    # u_mag = np.sqrt(u_plot**2 + v_plot**2)
    # plt.contourf(*meshgrid,u_mag,cmap='jet',levels = np.linspace(0,1,20))
    # plt.colorbar()
    # plt.show()
    
    # plt.contourf(*meshgrid,p,cmap='jet',levels = 100)
    # plt.colorbar()
    # plt.show()
    
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
        
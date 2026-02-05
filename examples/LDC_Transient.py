'''
Transient lid driven cavity for Re = 100

Using weakly compressible flow, we demonstrate the LDC problem as a time dependent flow.

The 2 fields to solve are the velocity and density (rho) fields.

Here we solve the conservative momentum equations where m is the target of interest rather than solving directly u as density is not a constant:

m_{i} = rho_{i}*u_{i}
p_{i} = cs**2(rho_{i} - rho_0)

m_{i+1} = m_{i} + dt*(-dp - u_conv + u_diff - u_damp)_{i}
rho_{i+1} = rho_{i} - dt*(div(m))

u_{i+1} = m_{i+1}/rho_{i+1}

Here:
- u_conv = div(m outer u)
- u_damp is biharmonic damping applied to all fluid nodes to prevent pressure checkerboarding

Geometry Params:
Domain = 1x1
n = 201
Runtime Params
U = 1.
Re = 100
rho_0 = 1.
M = 0.1
CFL_limit = 0.1
biharmonic_eps = 0.01

'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM import (Laplacian,
                            Grad,
                            GridBoundary,
                            Divergence,
                            scalarVectorMult,
                            OuterProduct)
from pde_module.time_step import ForwardEuler 
from pde_module.stencil import MapWise
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
    grid = Grid(dx = dx,num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params
    viscosity = 1/100
    density = 1.
    M = 0.1
    cs = U/M
    c2 = cs**2
    
    CFL_LIMIT = 0.1
    
    d_vis = CFL_LIMIT*float(dx**2/(viscosity))
    d_conv = CFL_LIMIT*dx/cs
    d_acoustic = CFL_LIMIT*dx/(cs + U)
    
    dt = min(d_vis,d_conv,d_acoustic)
    damping_eps = -0.01*(np.exp( 4*np.log(dx) - np.log(dt))/viscosity) # Use Logs for better numerical stability
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
    
    u_damping = Laplacian(2,dx,ghost_cells)
    
    rho = grid.create_node_field(1)
    rho.fill_(1.)
    rho_BC = GridBoundary(rho,dx,ghost_cells)
    rho_BC.vonNeumann_BC('ALL',0.)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)
    rho_step = ForwardEuler(rho.dtype)
    
    np.set_printoptions(precision=3,suppress= False)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)

    rho_step = ForwardEuler(rho.dtype)
    
    momentum_div = Divergence(2,u.shape,dx,ghost_cells)
    momentum = scalarVectorMult(2)

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
        u_damp = u_damping(u_diff,damping_eps)
        #Pressure Gradient
        p = get_p(rho_fix,c2,density)
        # print(p.numpy().squeeze())
        dp = p_grad(p,-1.)
        # Sum
        u_F = dp - u_conv + u_diff + u_damp
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
      
    step_per_frame = 750
    ani = FuncAnimation(fig,render , frames= 300, interval=50, repeat=False,fargs = [step_per_frame,dt])
        
    # step_per_frame = 1000
    # ani = FuncAnimation(fig,render , frames= 125, interval=50, repeat=False,fargs = [step_per_frame,dt])
    # ani.save('LDC_Transient.gif', writer='ffmpeg', fps=30,dpi=60)
    plt.show()
   
    meshgrid,us = grid.get_plotting_for('node',u)
    meshgrid,p = grid.get_plotting_for('node',rho)
    u_plot = us[:,:,0]
    v_plot = us[:,:,1]
    
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
        
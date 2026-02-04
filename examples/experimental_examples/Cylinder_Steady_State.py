'''
Transient Flow Around A Cylinder Re = 20

Using the ACM method, we demonstrate flow across a cylinder. ACM essentially is weakly compressible + increasing Mach number of the flow to quickly reach steady state

The 2 fields to solve are the velocity and pressure fields.

Here we assume density is constant and solve the following:

u_{i+1} = u_{i} + dt*(-dp - u_conv + u_diff + u_farfield)_{i}
p_{i+1} = p_{i} - dt*beta*(div(u))

Here:
- u_conv = du*u where du is grad(u) (a matrix)
- u_farfield is farfield damping applied to the outer 15 rows/columns to absorb incoming waves

Geometry Params:
H,L = 10x40
D = 1
n = 201
Runtime Params
U = 1.
Re = 100
rho_0 = 1.
beta = 0.3
CFL_limit = 0.8
farfield_sigma_max = 0.2

NOTE:
That at ~ Re =40, The flow transitions into unsteady state flow (karman vortex) so this is not suitable for high Re steady state flow.
'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry import Grid
from pde_module.FDM import Laplacian,Grad,GridBoundary,Divergence,DampingLayer,ImmersedBoundary
from pde_module.time_step import ForwardEuler 
from warp.types import vector
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 201
    H = 10
    L = 10
    
    dx = H/(n-1)
    
    ghost_cells = 1
    U = 1.
    D = 1.
    R = D/2
    centre = (L/2,H/2)
    grid = Grid(dx = dx,num_points=(4*n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params 
    viscosity = 1/20 # Equiv to 1/Re
    beta = 0.3
    CFL_LIMIT = 0.5
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    print(dt)
    farfield_sigma_max = 0.2/dt
    cyl = lambda x,y,z : (x-centre[0])**2 + (y-centre[1])**2 <= R**2
    
    meshgrid = grid.create_meshgrid('node')
    # Define Modules
    U_ref = wp.vec2f([U,0.])
    p_ref = vector(1,float)(0.)
    
    u = grid.create_node_field(2)
    u.fill_(U_ref)
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.vonNeumann_BC('ALL',0.)
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
    
    u_damping = Laplacian(2,dx,ghost_cells)
    
    u_div = Divergence(2,u.shape,dx,ghost_cells)
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
    
    u_farfield = DampingLayer(2,15,2.,u.shape,dx,ghost_cells)
    p_farfield =DampingLayer(1,15,2.,p.shape,dx,ghost_cells)
    
    
    t= 0

    np.set_printoptions(precision=1,suppress= False)
    
    def f(t,u,p):
        u_ibm = u_cyl(u)
        # print(u_ibm.numpy().squeeze()[:,:,0])
        u_fix = u_BC(u_ibm)
        u_diff = u_lapl(u_fix,viscosity)
        du = u_grad(u_fix)
        u_incomp = u_div(u_fix,-beta)
        u_conv = du*u
        
        #Pressure Field Calcs
        p_ibm = p_cyl(p)
        p_fix = p_BC(p_ibm)
        dp = p_grad(p_fix,-1.)
        
        # p_far = p_farfield(p,p_ref,farfield_sigma_max)
        u_far = u_farfield(u,U_ref,farfield_sigma_max)
        
        u_F = dp - u_conv + u_diff + u_far
        u_next = u_step(u_fix,u_F,dt)
        p_next = p_step(p_fix,u_incomp,dt)
        u,p = u_next,p_next
        
        return u,p
    
    u,p = f(t,u,p)
    t+=dt
    
    # assert False
    with wp.ScopedCapture() as capture:
        u,p = f(t,u,p)
    
    meshgrid = grid.create_meshgrid('node')[:2]
    X,Y = [m.squeeze() for m in meshgrid]
    fig, ax = plt.subplots()
    im = ax.imshow(u.numpy().squeeze()[:,:,0].T,origin='lower', animated=True, cmap='jet',vmin = 0.,vmax = 1.5*U)
    # im = ax.contourf(X,Y,,cmap= 'jet')
    fig.colorbar(im, ax=ax, label='U mag')
    
    step_per_frame = 100
    def render(frame,step_per_frame,dt):
        
        for i in range(step_per_frame):
            wp.capture_launch(capture.graph)
        
        t = frame*step_per_frame*dt
        new_u = u.numpy().squeeze()
        u_mag = (new_u[:,:,0]**2 + new_u[:,:,1]**2)**0.5
        ax.set_title(f'Step {frame*step_per_frame} Time = {t:.3f} Re = {1/viscosity:.1F}')
        # step += 1
        im.set_data(u_mag.T)
        return [im]
        
    
    ani = FuncAnimation(fig,render , frames= 500, interval=1, repeat=False,fargs = [step_per_frame,dt])
    plt.show()
    
        
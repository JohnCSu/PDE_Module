'''
Transient Flow Around A Cylinder Re = 100

Using weakly compressible flow, we demonstrate the classic vortex shedding over a cylinder. 

The 2 fields to solve are the velocity and density (rho) fields.

Here we solve the conservative momentum equations where m is the target of interest rather than solving directly u as density is not a constant:

m_{i} = rho_{i}*u_{i}
p_{i} = cs**2(rho_{i} - rho_0)

m_{i+1} = m_{i} + dt*(-dp - u_conv + u_diff - u_damp + u_farfield)_{i}
rho_{i+1} = rho_{i} - dt*(div(m))

u_{i+1} = m_{i+1}/rho_{i+1}

Here:
- u_conv = div(m outer u)
- u_damp is biharmonic damping applied to all fluid nodes to prevent pressure checkerboarding
- u_farfield is farfield damping applied to the outer 15 rows/columns to absorb incoming waves

Geometry Params:
H,L = 12x48
R = 1
n = 201
Runtime Params
U = 1.
Re = 100
rho_0 = 1.
M = 0.1
CFL_limit = 0.1
biharmonic_eps = 0.02
farfield_sigma_max = 0.2

Features Used:
- Staircase Approximation For Immersed Boundary Condition (First Order Accuracy)
- DampingLayers for farfield conditions and absorb both acoustice and velocity waves
- Biharmonic Damping to reduce pressure checkerboarding

'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry import Grid
from pde_module.FDM import (Laplacian,
                            Grad,
                            GridBoundary,
                            Divergence,
                            DampingLayer,
                            ImmersedBoundary,
                            scalarVectorMult,
                            OuterProduct)
from pde_module.time_step import ForwardEuler 
from pde_module.stencil import MapWise
from warp.types import vector
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 201
    H = 12
    L = 12
    
    dx = H/(n-1)
    
    ghost_cells = 1
    U = 1.
    D = 1
    R = D/2
    centre = (L/2,H/2)
    grid = Grid(dx = dx,num_points=(4*n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params 
    viscosity = 1/100 # Equiv to 1/Re
    density = 1.
    M = 0.1
    cs = U/M
    c2 = cs**2
    
    # beta = 0.3
    CFL_LIMIT = 0.1
    
    d_vis = CFL_LIMIT*float(dx**2/(viscosity))
    d_conv = CFL_LIMIT*dx/cs
    d_acoustic = CFL_LIMIT*dx/(cs + U)
    
    # print(beta)
    dt = min(d_vis,d_conv,d_acoustic)
    
    damping_eps = -0.02*(np.exp( 4*np.log(dx) - np.log(dt))/viscosity) # Use Logs for better numerical stability
    farfield_sigma_max = 0.3/dt
    
    print(dt)
    cyl = lambda x,y,z : (x-centre[0])**2 + (y-centre[1])**2 <= R**2
    # square = lambda x,y,z: np.maximum(np.abs(x-centre[0]),np.abs(y-centre[1])) <= R
    
    meshgrid = grid.create_meshgrid('node')
    
    u = grid.create_node_field(2)
    
    U_ref = [U,0.]
    u.fill_(U_ref)
    
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.vonNeumann_BC('ALL',0.)
    u_BC.dirichlet_BC('-X',U,0)
    u_BC.dirichlet_BC('-X',0.1,1)
    u_BC.dirichlet_BC('+Y',U,0)
    u_BC.dirichlet_BC('+Y',U_ref[1],1)
    u_BC.dirichlet_BC('-Y',U,0)
    u_BC.dirichlet_BC('-Y',U_ref[1],1)
    
    
    u_BC.vonNeumann_BC('+X',0.)
    
    u_cyl = ImmersedBoundary(u,dx,ghost_cells)
    u_cyl.from_bool_func(cyl,meshgrid)
    u_cyl.finalize()
    u_cyl.dirichlet_BC('ALL',0.)
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    
    u_div = Divergence(2,u.shape,dx,ghost_cells)
    u_step = ForwardEuler(u.dtype)
    
    u_outer = OuterProduct(2,2)
    u_conv_div = Divergence((2,2),u.shape,dx,ghost_cells)
    
    u_damping = Laplacian(2,dx,ghost_cells)
    
    rho = grid.create_node_field(1)
    rho.fill_(1.)
    
    # print(rho.numpy().squeeze())
    rho_BC = GridBoundary(rho,dx,ghost_cells)
    rho_BC.vonNeumann_BC('ALL',0.)
    rho_BC.dirichlet_BC('+X',density)
    
    p_grad = Grad(1,rho.shape,dx,ghost_cells)
    rho_step = ForwardEuler(rho.dtype)
    
    np.set_printoptions(precision=3,suppress= False)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)

    rho_step = ForwardEuler(rho.dtype)
    
    momentum_div = Divergence(2,u.shape,dx,ghost_cells)
    momentum = scalarVectorMult(2)
    
    rho_cyl = ImmersedBoundary(rho,dx,ghost_cells)
    rho_cyl.from_bool_func(cyl,meshgrid)
    rho_cyl.finalize()
    rho_cyl.vonNeumann_BC('ALL',0.)
    
    print(np.all(rho_cyl.boundary_type == 2))
    @wp.func
    def get_p_op(rho:vector(1,float),c2:float,density:float):
        rho[0] = c2*(rho[0] - density)
        return rho
    
    get_p = MapWise(get_p_op)
    get_u_from_m = MapWise(lambda m,rho: m/rho[0])
    p_grad = Grad(1,rho.shape,dx,ghost_cells)
    
    U_ref = wp.vec2f(U_ref)
    rho_ref = vector(1,float)(density)
    u_farfield = DampingLayer(2,15,2.,u.shape,dx,ghost_cells)
    rho_farfield =DampingLayer(1,15,2.,u.shape,dx,ghost_cells)
    
    np.set_printoptions(precision=1,suppress= False)
    t= 0.
    
    
    
    print(damping_eps)
    def f(t,u,rho):
        u_ibm = u_cyl(u,fill_value = 0)
        rho_ibm = rho_cyl(rho,fill_value = density)
        # u_ibm = u
        # rho_ibm = rho 
        u_fix = u_BC(u_ibm)
        rho_fix = rho_BC(rho_ibm)
        # print(u_fix.numpy().squeeze()[:,:,0])
        m = momentum(rho_fix,u_fix)
        m_div = momentum_div(m,-1.)
        
        #Convec + diff
        u_diff = u_lapl(u_fix,viscosity)        
        u_2 = u_outer(m,u_fix)
        u_conv = u_conv_div(u_2)
        
        u_damp = u_damping(u_diff,damping_eps)
        
        p = get_p(rho_fix,c2,density)
        dp = p_grad(p,-1.)
        
        u_far = u_farfield(u_fix,U_ref,farfield_sigma_max)
        rho_far = rho_farfield(rho_fix,rho_ref,farfield_sigma_max)
        
        # Sum
        u_F = dp - u_conv +u_diff + u_far + u_damp
        m_next = u_step(m,u_F,dt)    
        rho_next = rho_step(rho_fix,m_div +rho_far,dt) #
      
        u_next = get_u_from_m(m_next,rho_next)
        u,rho = u_next,rho_next
        return u,rho
    
    u,rho = f(t,u,rho)
    t+=dt
    # exit()
    # assert False
    with wp.ScopedCapture() as capture:
        u,rho = f(t,u,rho)
    
    meshgrid = grid.create_meshgrid('node')[:2]
    X,Y = [m.squeeze() for m in meshgrid]
    fig, ax = plt.subplots()
    im = ax.imshow(u.numpy().squeeze()[:,:,0].T,origin='lower', animated=True, cmap='jet')
    # im = ax.contourf(X,Y,,cmap= 'jet')
    col = fig.colorbar(im, ax=ax, label='U mag')
    
    def render(frame,step_per_frame,dt):
        t = frame*step_per_frame*dt
        
        for i in range(step_per_frame):
            wp.capture_launch(capture.graph)
            # u,rho = f(t,u,rho)
        
        new_u = u.numpy().squeeze()
        u_mag = (new_u[:,:,0]**2 + new_u[:,:,1]**2)**0.5
        ax.set_title(f'Step {frame*step_per_frame} Time = {t:.3f} Max: {u_mag.max()}')
        im.set_data(u_mag.T)
        
        # new_rho = rho.numpy().squeeze()
        # ax.set_title(f'Step {frame*step_per_frame} Time = {t:.3f} Max: {new_rho.max()}')
        # im.set_data(new_rho.T)
        
        # step += 1
        # 
        im.set_clim(vmin=0, vmax=1.5*U)
        return [im]
        
    step_per_frame = 400
    ani = FuncAnimation(fig,render , frames= 1000, interval=1, repeat=False,fargs = [step_per_frame,dt])

    plt.show()
    print('hi')
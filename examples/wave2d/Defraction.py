'''
2D Wave Equation Diffraction Example Showing Viscous Damping Layer and transient BC

du**2/dt**2 = c*laplace(u)

We solve the wave equation as a system of first order PDEs

if u is displacement and v is velocity (time derivitive of u):

du/dt = v
dv/dt = c*laplace(u)

Over a 2D Grid. We also have a slot



For Initial Condition:
u(x,y,0) = A*(np.sin(m*np.pi*x/L)*np.sin(m*np.pi*y)/L)

- A: Amplitude = 1
- m: number of modes along each each = 2
- L: Length along axis  = 1
- c: Wave Speed = 1

With Boundary Condition at the grid perimeter:

At (0,y): sin(omega*t) for t < 2. then 0 afterwards
where omega = 40.*wp.pi/(W)
 
For all other BC, a  Dirichlet BC of 0. is applied

A Viscous damping layer proportional to wave velocity is applied 10-15 nodes thick to absorb waves and act as a decaying farfield condition


'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM.laplacian import Laplacian
from pde_module.time_step.forwardEuler import ForwardEuler
from pde_module.FDM import ImmersedBoundary,GridBoundary,ViscousDampingLayer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from warp.types import vector

wp.init()

@wp.func
def sin_wave(current_values:wp.array3d(dtype = vector(1,float)),
             nodeID:wp.vec3i,
             varID:int,
             coordinates:wp.array3d(dtype=vector(3,float)),
             t:float,
             dx:float,
             omega:float):
    
    if t < 2.:
        return wp.sin(omega*t)
    else:
        return 0.
    

if __name__ == '__main__':
    #Geometry
    n = 61
    L = 1
    dx = L/(n-1)
    
    m = 2
    
    #Params
    A = 1.
    c = 0.1
    c2 = c**2
    # dt = float(dx**2/(4*alpha))
    CFL_limit = 0.3
    
    dt = CFL_limit*float(1/(c*np.sqrt(2./dx**2)))
    print(dt)
    ghost_cells = 1
    
    grid = Grid(dx = 1/(n-1),num_points=(n*m,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    W,H = m*L,L
    
    u0 =grid.create_node_field(1)
    v0 = grid.create_node_field(1)
    # Define Modules
    
    slot_BC = ImmersedBoundary(u0,dx,ghost_cells)
    
    def slot(x,y,z):
        
        x_bool = np.logical_and(x >= 0.5, x<= 0.55)
        y_bool = np.logical_or(y >= 0.55,y <= 0.45)
        # print(0.45 <= x <= 0.55)
        return np.logical_and(x_bool, y_bool ) 
    
    slot_BC.from_bool_func(slot,grid.create_meshgrid('node'))
    slot_BC.finalize()
    slot_BC.dirichlet_BC('ALL',0.)
    slot_BC.show_bitmask()
    
    BC = GridBoundary(u0,dx,ghost_cells,grid_coordinates=grid.node_coordinates)
    BC.dirichlet_BC('ALL',0.)
    # BC.dirichlet_BC('+X',0.)
    BC.dirichlet_BC('-X',sin_wave,0)
    
    damping_thickness = 15
    u_damp = ViscousDampingLayer(1,damping_thickness,u0.shape,dx,ghost_cells)
    u_damp.exclude('-X')
    C_max = u_damp.calculate_C_max(c,dx*damping_thickness,u_damp.p,1e-3)
    # C_max = 0.4
    print(C_max, C_max*dt)
    u_step = ForwardEuler(u0.dtype)
    v_step = ForwardEuler(v0.dtype)
    lapl = Laplacian(1,dx,ghost_cells)
    
    np.set_printoptions(precision=2,suppress= False,linewidth= 200)
    def generator():
        u,v =  u0,v0
        k = 40.*wp.pi/(W)
        omega = wp.float32(k*c)
        t = 0.
        for i in range(3000):        
            #Boundary conditions
            u3 = BC(u,t,{'-X': omega})
            # u2= u3
            u2 = slot_BC(u3)
            # Calculate Laplacian 
            stencil =lapl(u2,c2)
            u_far = u_damp(v,C_max)
            # Time Step
            v_next = v_step(v,stencil+u_far,dt)
            u_next = u_step(u2,v,dt)
            u,v = u_next,v_next
            t += dt
            
            if i % 5 ==0:
                yield i,u,v

                

meshgrid = grid.create_meshgrid('node')
X,Y,_ = [m.squeeze() for m in meshgrid]


fig, ax = plt.subplots()
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
im = ax.imshow(u0.numpy().squeeze().T,origin='lower',alpha = 1., animated=True, cmap='jet',vmin = -1,vmax = 1)
# im = ax.contourf(X,Y,,cmap= 'jet')
fig.colorbar(im, ax=ax, label='U mag')

x_plot = np.linspace(0,L,n)


def render(frame):
    # ax.clear()
    step,us,vs = frame
    ax.set_title(f'Step {step} ')
    us = us.numpy().squeeze()
    us = us + slot_BC.bitmask.squeeze()*2
    im.set_data(us.T)
    return im


ani = FuncAnimation(fig,render , frames= generator(), interval=50, repeat=False)
plt.show()
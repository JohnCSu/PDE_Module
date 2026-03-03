'''
2D Wave Equation Example Showing Usage of transient BCs

du**2/dt**2 = c*laplace(u)

We solve the wave equation as a system of first order PDEs

if u is displacement and v is velocity (time derivitive of u):

du/dt = v
dv/dt = c*laplace(u)

Over a 2D 1x1 Grid

For Initial Condition:
u(x,y,0) = A*(np.sin(m*np.pi*x/L)*np.sin(m*np.pi*y)/L)

- A: Amplitude = 1
- m: number of modes along each each = 2
- L: Length along axis  = 1
- c: Wave Speed = 1

With Boundary Condition at the grid perimeter:

At (0,y): sin(omega*t)
where omega = 8.*wp.pi/(W)
At (W,y): dirichlet BC of 0 
For all other BC, a  Von Neumann BC of 0. is applied


'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM.laplacian import Laplacian
from pde_module.time_step.forwardEuler import ForwardEuler
from pde_module.FDM.boundary.gridBoundary import GridBoundary
from pde_module.FDM import ImmersedBoundary
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
    return wp.sin(omega*t)
    

if __name__ == '__main__':
    #Geometry
    n = 31
    L = 1
    dx = L/(n-1)
    
    #Params
    A = 1.
    c = 2.
    c2 = c**2
    # dt = float(dx**2/(4*alpha))
    CFL_limit = 0.7
    
    dt = float(1/(c*np.sqrt(2./dx**2)))
    print(dt)
    ghost_cells = 1
    
    grid = Grid(dx = 1/(n-1),num_points=(n*10,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    W,H = 10*L,L
    
    u0 =grid.create_node_field(1)
    v0 = grid.create_node_field(1)
    # Define Modules
    
    BC = GridBoundary(u0,dx,ghost_cells,grid_coordinates=grid.node_coordinates)
    BC.vonNeumann_BC('ALL',0.)
    BC.dirichlet_BC('+X',0.)
    BC.dirichlet_BC('-X',sin_wave,0)
    
    u_step = ForwardEuler(u0.dtype)
    v_step = ForwardEuler(v0.dtype)
    lapl = Laplacian(1,dx,ghost_cells)
    
    
    def generator():
        u,v =  u0,v0
        k = 8.*wp.pi/(W)
        omega = wp.float32(c*k)
        t = 0.
        for i in range(2000):        
            u2 = BC(u,t,{'-X': omega})
            stencil =lapl(u2,c2)
            v_next = v_step(v,stencil,dt)
            u_next = u_step(u2,v,dt)
            u,v = u_next,v_next
            t += dt
            if i % 10 ==0:
                yield i,u,v
                

meshgrid,us = grid.get_plotting_for('node',u0)
X,Y = [m.squeeze() for m in meshgrid]
fig, ax = plt.subplots()
im = ax.imshow(u0.numpy().squeeze().T,origin='lower', animated=True, cmap='jet',vmin = -A,vmax = A)
# im = ax.contourf(X,Y,,cmap= 'jet')
fig.colorbar(im, ax=ax, label='U mag')

x_plot = np.linspace(0,L,n)


def render(frame):
    step,us,vs = frame
    ax.set_title(f'Step {step} ')
    # step += 1
    im.set_data(us.numpy().squeeze().T)
    return [im]


ani = FuncAnimation(fig,render , frames= generator(), interval=50, repeat=False)
plt.show()
'''
2D Wave Equation Example For a standing wave

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
u(x_b,y_b,t) = 0
'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM.laplacian import Laplacian
from pde_module.time_step.forwardEuler import ForwardEuler
from pde_module.FDM.boundary.gridBoundary import GridBoundary
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
wp.init()

if __name__ == '__main__':
    #Geometry
    n = 101
    L = 1
    dx = L/(n-1)
    
    #Params
    m = 2
    A = 1.
    c = 1.
    c2 = c**2
    # dt = float(dx**2/(4*alpha))
    CFL_limit = 0.7
    
    dt = 1/(c*np.sqrt(2./dx**2))
    print(dt)
    ghost_cells = 1
    
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    IC = lambda x,y,z: A*(np.sin(m*np.pi*x/L)*np.sin(m*np.pi*y)/L)
    
    u0 =grid.initial_condition('node',IC)
    v0 = grid.create_node_field(1)
    # Define Modules
    
    BC = GridBoundary(u0,dx,1)
    BC.dirichlet_BC('ALL',0.)
    
    u_step = ForwardEuler(u0.dtype)
    v_step = ForwardEuler(v0.dtype)
    lapl = Laplacian(1,dx,ghost_cells)
    
    def generator():
        u,v =  u0,v0
        for i in range(2001):        
            u2 = BC(u)
            stencil =lapl(u2,c2)
            v_next = v_step(v,stencil,dt)
            u_next = u_step(u2,v,dt)
            u,v = u_next,v_next
            if i % 5 ==0:
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
    ax.set_title(f'Standing Wave Step {step} Time {step*dt:.3F}')
    # step += 1
    im.set_data(us.numpy().squeeze().T)
    return [im]


ani = FuncAnimation(fig,render , frames= generator(), interval=50, repeat=False)
plt.show()
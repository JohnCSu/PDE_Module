import numpy as np
import warp as wp


from pde_module.experimental.grid import Grid
from pde_module.experimental.laplacian import Laplacian
from pde_module.experimental.time_integrators import ForwardEuler

wp.init()
wp.config.mode = "debug"

if __name__ == '__main__':
    n = 3
    L = 1
    dx = L/(n-1)
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.))
    
    
    u = grid.create_node_field(1,1)
    
    
    
    lapl = Laplacian(1,grid.dx)
    lapl.setup(u)
    
    euler_step = ForwardEuler(u.dtype)
    
    
    stencil =lapl(u)
    u_next = euler_step(u,stencil,0.1)
    
    
    
    # IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    # initial_value =grid.initial_condition(IC)
    
    t = 0
    
    dx = grid.dx
    dt = float(dx**2/(4*0.1))

    
    # boundary = GridBoundary(grid,1,dynamic_array_alloc= False)
    # boundary.dirichlet_BC('ALL',0.)
    
    # laplcian_stencil = Laplacian(grid,1,dynamic_array_alloc=False)
    # time_step = ForwardEuler(grid,1,dynamic_array_alloc= False)
    
    # # with wp.ScopedCapture(device="cuda") as iteration_loop:
    # np.set_printoptions(precision=2)
    # input_values = initial_value
    # print(input_values.numpy()[0,:,:,0,0])
    
    # for i in range(1001):
    #     boundary_corrected_values = boundary(input_values)
    #     # print(boundary_corrected_values.numpy()[0,:,:,0,0])        
    #     laplace = laplcian_stencil(boundary_corrected_values,scale =0.1)
    #     new_value = time_step(boundary_corrected_values,laplace,dt)
    #     input_values = new_value

    #     t += dt
    #     if t > 1.:
    #         break
        
    # print(f't = {t:.3e} max value = {np.max(input_values.numpy().max()):.3E}, dt = {dt:.3E}')
    
    # to_plot = grid.trim_ghost_values(input_values)
    # u = to_plot[0,:,:,0]
    # meshgrid = grid.meshgrid('node',False,for_plotting=True)
    # meshgrid = [m.T for m in meshgrid]
    # # print(f'max u {np.max(u):.3E}')
    
    # plt.quiver(*meshgrid[::-1],u.T,v.T)
    # plt.show()
    
    
    # plt.contourf(*meshgrid[::-1],u.T,cmap ='jet',levels = 100)
    # plt.colorbar()
    # plt.show()
    
    
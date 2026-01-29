
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.experimental.grid import Grid
from pde_module.experimental.FDM.laplacian import Laplacian
from pde_module.experimental.time_integrators import ForwardEuler
from pde_module.experimental.FDM.gridBoundary import GridBoundary
from pde_module.experimental.FDM.grad import Grad
from pde_module.experimental.FDM.divergence import Divergence


wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 101
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)

    # u = grid.create_node_field(1,1)

    IC = lambda x,y,z: (np.sin(np.pi*x)*np.sin(np.pi*y))
    u =grid.initial_condition('node',IC)
    
    # meshgrid = grid.create_meshgrid('node')
    
    # plot_meshgrid = [m.squeeze() for m in meshgrid]
    # plt.contourf(*plot_meshgrid[:2],u.squeeze())
    # plt.show()
    
    # dx = grid.dx
    alpha = 0.1
    dt = float(dx**2/(4*alpha))
    
    # Define Modules
    BC = GridBoundary(u,dx,1)
    BC.dirichlet_BC('ALL',0.)
    
    
    euler_step = ForwardEuler(u.dtype)
    
    grad = Grad(1,u.shape,dx,ghost_cells=ghost_cells)
    div = Divergence('vector',u.shape,dx,ghost_cells)
    lapl = Laplacian(1,dx,ghost_cells)

    lapl.setup(u)
    # div.setup(u)
    # grad.setup(u)
    BC.setup(u)
    
    
    t= 0
    for i in range(1000):
        u2 = BC(u)
        # stencil =lapl(u2,alpha)
        
        g = grad(u)
        assert g.dtype._length_ == 2
        stencil = div(g,alpha)
        
        u_next = euler_step(u2,stencil,dt)
        u = u_next
        t+= dt

        if i % 100 == 0:
            print(f'Max u = {u.numpy().max()} at t = {t},iter = {i}')
    
        

# 
# 

# # print(u.shape)
# # print(u.numpy().squeeze())


# print(BC.boundary_interior.squeeze())
# print(u2.numpy().squeeze())




# t = 0




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


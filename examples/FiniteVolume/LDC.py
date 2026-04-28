from pde_module.FV import (Diffusion,
                           Advection,
                           Divergence,
                           FiniteVolumeMesh,
                           Grad,
                           Boundary,
                           flags)
from pde_module.mesh.mesh_generators import create_Uniform_grid
from pde_module.mesh import to_pyvista
from pde_module.time_step import ForwardEuler
import numpy as np
import warp as wp
import pyvista as pv
import matplotlib.pyplot as plt

wp.init()
wp.set_module_options({"enable_backward": False})

N = 101
nodes_per_axis = (N,N,2)
dx = 1/(N-1)
L = (dx*(N-1),dx*(N-1),dx)

nodes,cells_array,cell_types_array = create_Uniform_grid(dx = dx,nodes_per_axis = nodes_per_axis)
FV_mesh = FiniteVolumeMesh(nodes,cells_array,cell_types_array)

# Set Groups for faces
for i,axis in enumerate(['X','Y','Z']):
    for j,sign in enumerate(['-','+']):
        coords = FV_mesh.exterior_faces.centroids[:,i] 
        x = 0 if j == 0 else L[i]
        key = sign+axis
        FV_mesh.exterior_faces.groups[key] = np.argwhere(np.isclose(coords,x)).ravel()

# Fields and params
u = FV_mesh.create_cell_field(3)
p = FV_mesh.create_cell_field(1)
density = 1.
viscosity = 1/100
beta = 0.3
CFL_LIMIT = 0.5
dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
t = 0
#Modules
FV_mesh.to_warp()
FV_mesh.verify_fp(print_output=True,ignore_int = True)

u_time_step = ForwardEuler()
p_time_step = ForwardEuler()

u_boundary = Boundary(FV_mesh,3)
u_laplace = Diffusion(FV_mesh)
u_convection = Advection(FV_mesh)
u_divergence = Divergence(FV_mesh)

p_boundary = Boundary(FV_mesh,1)
p_grad = Grad(FV_mesh)

# Set Boundary Conditions
u_boundary.set_BC('ALL',flags.DIRICHLET,0)
u_boundary.set_BC('-Z',flags.VON_NEUMANN,0)
u_boundary.set_BC('+Z',flags.VON_NEUMANN,0)
u_boundary.set_BC('+Y',flags.DIRICHLET,1.,0) # U = (1,0)
p_boundary.set_BC('ALL',flags.VON_NEUMANN,0)


def f(t,u,p):
    u_BC = u_boundary(u)
    diff = u_laplace(u,u_BC,viscosity)
    conv = u_convection(u,u_BC,u,u_BC,density=density)
    div = u_divergence(u,u_BC,alpha = -beta)
    
    p_BC = p_boundary(p)
    del_p = p_grad(p,p_BC,alpha = -1/density)
    
    u_F = del_p + diff - conv
    
    u_next = u_time_step(u,u_F,dt)
    p_next = p_time_step(p,div,dt)
    wp.copy(u,u_next)
    wp.copy(p,p_next)
    return u,p

u,p = f(t,u,p)
with wp.ScopedCapture() as capture:   
    u,p = f(t,u,p)

for i in range(10001):
    wp.capture_launch(capture.graph)
    t+= dt
    if i % 1000 == 0:
        print(f't = {t:.3E},iter = {i},u_max {u.numpy()[0].max()}')

pv_mesh = to_pyvista(FV_mesh)

us = u.numpy()
u_plot = us[0,:]
v_plot = us[1,:]
u_mag = np.sqrt(u_plot**2 + v_plot**2)
pv_mesh.cell_data['U_mag'] = u_mag
pv_mesh.cell_data['U velocity'] = u_plot
pv_mesh.cell_data['V velocity'] = v_plot

plotter = pv.Plotter()
plotter.add_mesh(pv_mesh,scalars ='U_mag',show_edges = False, cmap= 'jet',clim = [0,1])
plotter.view_xy()
plotter.show()


import pandas as pd
v_benchmark = pd.read_csv(r'examples/v_velocity_results.csv',sep = ',')
u_benchmark = pd.read_csv(r'examples/u_velocity_results.txt',sep= '\t')

horizontal_line = pv_mesh.sample_over_line((0,L[1]/2,0),(L[0],L[1]/2,L[2]/2),resolution= N)
v_05 = horizontal_line.point_data['V velocity']

print(f"CFD max {v_05.max()}, Benchmark Max :{v_benchmark['100'].max()}")
plt.plot(v_benchmark['%x'],v_benchmark['100'],'o',label = 'Ghia et al')
plt.plot(horizontal_line.points[:,0],v_05)
plt.show()


vertical_line = pv_mesh.sample_over_line((L[0]/2,0,L[2]/2),(L[0]/2,L[1],L[2]/2),resolution= N)
u_05 = vertical_line.point_data['U velocity']

print(f"CFD max {u_05.max()}, Benchmark Max :{u_benchmark['100'].max()}")
plt.plot(u_benchmark['%y'],u_benchmark['100'],'o',label = 'Ghia et al')
plt.plot(vertical_line.points[:,1],u_05)
plt.show()
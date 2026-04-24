from pde_module.FV import (Diffusion,
                           Advection,
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

wp.init()
wp.set_module_options({"enable_backward": False})

N = 101
nodes_per_axis = (N,N,2)
dx = 1/(N-1)
L = (dx*N,dx*N,dx)


nodes,cells_array,cell_types_array = create_Uniform_grid(dx = dx,nodes_per_axis = nodes_per_axis)
FV_mesh = FiniteVolumeMesh(nodes,cells_array,cell_types_array)

# Set Groups for faces
for i,axis in enumerate(['X','Y','Z']):
    for j,sign in enumerate(['-','+']):
        coords = FV_mesh.exterior_faces.centroids[:,i] 
        x = 0 if j == 0 else L[i]
        key = sign+axis
        FV_mesh.exterior_faces.groups[key] = np.argwhere(np.isclose(coords,x)).ravel()


#Modules
time_step = ForwardEuler()
u_boundary = Boundary(FV_mesh,3)
u_laplace = Diffusion(FV_mesh)
u_convection = Advection(FV_mesh)

p_boundary = Boundary(FV_mesh,1)
p_grad = Grad(FV_mesh)


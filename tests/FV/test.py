from pde_module.FV import (Diffusion,
                           Advection,
                           FiniteVolumeMesh,
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
def test():
    

    N = 7
    nodes_per_axis = (N,5,4)
    dx = 1/(N-1)
    L = (dx*N,dx*(5),dx*4) # Have variation in number of cells
    
    nodes,cells_array,cell_types_array = create_Uniform_grid(dx = dx,nodes_per_axis = nodes_per_axis)
    FV_mesh = FiniteVolumeMesh(nodes,cells_array,cell_types_array)
    time_step = ForwardEuler()

    # Set Groups for faces
    for i,axis in enumerate(['X','Y','Z']):
        for j,sign in enumerate(['-','+']):
            coords = FV_mesh.exterior_faces.centroids[:,i] 
            x = 0 if j == 0 else L[i]
            key = sign+axis
            FV_mesh.exterior_faces.groups[key] = np.argwhere(np.isclose(coords,x)).ravel()
            
            
    scalar = FV_mesh.create_cell_field(1)
    velocity = FV_mesh.create_cell_field(3)
    
    adv = Advection(FV_mesh)
    diff = Diffusion(FV_mesh)
    vel_BC = Boundary(3,FV_mesh)
    vel_BC.set_BC('ALL',flags.DIRICHLET,0)
    vel_BC.set_BC('-Z',flags.VON_NEUMANN,0)
    vel_BC.set_BC('+Z',flags.VON_NEUMANN,0)
    
    
    BC = Boundary(1,FV_mesh)
    BC.set_BC('ALL',flags.DIRICHLET,0)
    BC.set_BC('-Z',flags.VON_NEUMANN,0)
    BC.set_BC('+Z',flags.VON_NEUMANN,0)

    #Params
    alpha = 0.1
    dt = float(dx**2/(4*alpha))
    t=0

    scalar_BC = BC(scalar)
    laplace = diff(scalar,scalar_BC,alpha)
    u_BC = BC(scalar)
    
    vel_boundary = vel_BC(velocity)
    conv = adv(scalar,scalar_BC,velocity,vel_boundary,density = 1.)
    
    return True



print(test())


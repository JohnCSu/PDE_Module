from pde_module.FV import (Diffusion,
                           Advection,
                           FiniteVolumeMesh,
                           Grad,
                           Boundary,
                           Divergence,
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
    N = 4 # 3x3 cells
    nodes_per_axis = (N,N,2)
    # dx = 1/(N-1)
    dx = 1.
    L = (dx*(N-1),dx*(N-1),dx) # Have variation in number of cells
    
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
            
    # print(FV_mesh.exterior_faces.groups)
    scalar = FV_mesh.create_cell_field(1,IC = 1.5)
    velocity = FV_mesh.create_cell_field(3,IC = 1.5)
    
    adv = Advection(FV_mesh)
    diff = Diffusion(FV_mesh)
    grad = Grad(FV_mesh)
    div = Divergence(FV_mesh)
    vel_adv = Advection(FV_mesh)
    vel_diff = Diffusion(FV_mesh)
    
    vel_BC = Boundary(FV_mesh,3)
    vel_BC.set_BC('ALL',flags.DIRICHLET,0)
    vel_BC.set_BC('-Z',flags.VON_NEUMANN,0)
    vel_BC.set_BC('+Z',flags.VON_NEUMANN,0)
    
    
    BC = Boundary(FV_mesh,1)
    BC.set_BC('ALL',flags.DIRICHLET,0)
    BC.set_BC('-Z',flags.VON_NEUMANN,0)
    BC.set_BC('+Z',flags.VON_NEUMANN,0)

    #Params
    alpha = 0.1
    dt = float(dx**2/(4*alpha))
    t=0

    pv_mesh = to_pyvista(FV_mesh)
    pl = pv.Plotter()
    pl.add_mesh(pv_mesh, color='white', show_edges=True,opacity = 0.2)
    pl.add_arrows(FV_mesh.faces.centroids,FV_mesh.faces.normals,mag = 0.4)
    pl.add_point_labels(FV_mesh.cells.centroids,np.arange(len(FV_mesh.cells)))
    # pl.show()
    scalar_BC = BC(scalar)
    scalar_grad = grad(scalar,scalar_BC,alpha = 1.)
    laplace = diff(scalar,scalar_BC,alpha)
    
    
    
    vel_boundary = vel_BC(velocity)
    vel_lapl = vel_diff(velocity,vel_boundary,alpha)
    
    
    
    scalar_conv = adv(scalar,scalar_BC,velocity,vel_boundary,density = 1.)
    vel_conv = vel_adv(velocity,vel_boundary,velocity,vel_boundary,density = 1.)
    
    u_div = div(velocity,vel_boundary) # Scalar
    
    # print(scalar_grad.numpy().squeeze()[:,0])
    # print(u_div.numpy().squeeze()[0])
    # print(vel_conv.numpy().squeeze()[:,0])
    
    assert all( np.isclose(x,4.5) for x in vel_conv.numpy().squeeze()[:,0])
    assert np.isclose(u_div.numpy().squeeze()[0] , 3.)
    assert np.allclose(scalar_grad.numpy().squeeze()[:,0],[1.5,1.5,0.])
    
    
    
    return True



print(test())


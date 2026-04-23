from pde_module.FV import Diffusion,FiniteVolumeMesh,Boundary,flags
from pde_module.mesh.mesh_generators import create_Uniform_grid
from pde_module.mesh import to_pyvista
from pde_module.time_step import ForwardEuler
import numpy as np
import warp as wp
import pyvista as pv

wp.init()


N = 101
nodes_per_axis = (N,N,2)
dx = 1/(N-1)
L = (dx*N,dx*N,dx)


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

IC = lambda centroids: (np.sin(centroids[:,0]*np.pi)*np.sin(centroids[:,1]*np.pi))[np.newaxis,:]

print(FV_mesh.cells.volumes)
print(np.linalg.vector_norm(FV_mesh.exterior_faces.normals,axis = 1,ord = 2))


# IC = lambda centroids: np.ones(shape = (1,len(centroids)),dtype=np.float32)
u = FV_mesh.create_cell_field(1,IC)

diff = Diffusion(FV_mesh)
BC = Boundary(1,FV_mesh)
BC.set_BC('ALL',flags.DIRICHLET,0)
BC.set_BC('-Z',flags.VON_NEUMANN,0)
BC.set_BC('+Z',flags.VON_NEUMANN,0)

#Params
alpha = 0.1
dt = float(dx**2/(4*alpha))
t=0

print(len(FV_mesh.exterior_faces))

with wp.ScopedTimer('Laplace',cuda_filter=wp.TIMING_ALL):
    for i in range(1000):
        u_BC = BC(u)
        laplace = diff(u,u_BC,alpha)
        u_out = time_step(u,laplace,dt)
        wp.copy(u,u_out)
        t += dt
        if i % 100 == 0:
            print(f'Max u = {u.numpy().max()} at t = {t},iter = {i}')

pv_mesh = to_pyvista(FV_mesh)
pv_mesh.cell_data['u'] = u.numpy().squeeze()
pl = pv.Plotter()
pl.add_mesh(pv_mesh,scalars = 'u', color='white', show_edges=False,cmap='jet',clim = [0,1])
# pl.show()





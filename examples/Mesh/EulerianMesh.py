import numpy as np
from pde_module.mesh import EulerianMesh
from pde_module.mesh.mesh_generators import generate_cube_mesh
from pde_module.mesh import to_pyvista
import pyvista as pv

if __name__ == "__main__":
    # mesh = generate_cube_mesh_simple(divisions= 1,cell_type='wedge')
    mesh = generate_cube_mesh(cell_type='hex',show_gui=False,divisions=(3,3,3))
    
    EulerMesh = EulerianMesh(mesh.nodes,mesh.cells.connectivity,mesh.cells.types)
    
    # print(EulerMesh.faces.ownerNeighbor)
    # print(EulerMesh.cells.volumes)
    print(EulerMesh.cells.neighbors)
    print(EulerMesh.cells.neighbors_offset)
    
    
    middle_cell_offset = EulerMesh.cells.neighbors_offset[13]
    num_neighbors = EulerMesh.cells.neighbors[middle_cell_offset]
    print(EulerMesh.cells.neighbors[middle_cell_offset:middle_cell_offset+num_neighbors+1])
    
    assert num_neighbors == 6
     
    
    cell_ids = np.arange(len(EulerMesh.cells))
    # assert len(mesh.cells) == 2
    pv_mesh = to_pyvista(EulerMesh)
    
    edges = pv_mesh.extract_feature_edges(boundary_edges=True,
                                    manifold_edges=True,
                                    non_manifold_edges=True,
                                    feature_edges=False)
    pl = pv.Plotter()
    pl.add_mesh(pv_mesh, color='white', show_edges=True,opacity = 0.2)
    pl.add_arrows(EulerMesh.faces.centroids,EulerMesh.faces.normals,mag = 0.4)
    pl.add_point_labels(EulerMesh.cells.centroids,cell_ids)
    
    # pl.add_mesh(edges, color='black', line_width=1)
    pl.show()
    
    
    
import numpy as np
from pde_module.mesh import Mesh
from pde_module.mesh.gmsh import generate_cube_mesh


if __name__ == "__main__":
    from pde_module.mesh import to_pyvista
    import pyvista as pv
    # mesh = generate_cube_mesh_simple(divisions= 1,cell_type='wedge')
    mesh = generate_cube_mesh(cell_type='wedge',show_gui=False)
    print(mesh.cells.unique_cell_types)
    
    pv_mesh = to_pyvista(mesh)
    
    edges = pv_mesh.extract_feature_edges(boundary_edges=True,
                                    manifold_edges=True,
                                    non_manifold_edges=True,
                                    feature_edges=False)
    pl = pv.Plotter()
    # pl.add_mesh(pv_mesh, color='white', show_edges=False,opacity = 0.1)
    pl.add_mesh(edges, color='black', line_width=1)
    pl.show()
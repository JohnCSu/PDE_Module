from pde_module.mesh.utils import generate_vectorized_vtk_hex
from pde_module.mesh.cell_types import HEX,TETRA,WEDGE
import numpy as np
from pde_module.mesh import Mesh,to_pyvista
if __name__ == '__main__':
    hex_nodes, vtk_connectivity = generate_vectorized_vtk_hex(1,1,2)
    
    tet_coords = np.array([
    [0.0, 0.0, 0.0], # Node 0: Origin
    [1.0, 0.0, 0.0], # Node 1: X-axis
    [0.0, 1.0, 0.0], # Node 2: Y-axis
    [0.0, 0.0, 1.0]  # Node 3: Apex (Z-axis)
    ])
    
    tet_coords[:,0] += 2.
    
    wedge_coords = np.array([
    # Bottom Triangle (z=0)
    [0.0, 0.0, 0.0], # Node 0
    [1.0, 0.0, 0.0], # Node 1
    [0.0, 1.0, 0.0], # Node 2
    # Top Triangle (z=1)
    [0.0, 0.0, 1.0], # Node 3
    [1.0, 0.0, 1.0], # Node 4
    [0.0, 1.0, 1.0]  # Node 5
],np.float32)
    
    # wedge_coords[:,2] += 2.
    
    tet_connectivity = np.array([0,0, 1, 2, 3]) + len(hex_nodes)
    tet_connectivity[0] = 4
    
    wedge_connectivity = np.array([0,0, 1, 2, 3, 4, 5]) + len(hex_nodes) + len(tet_coords)
    wedge_connectivity[0] = 6
    
    nodes = np.concat((hex_nodes,tet_coords,wedge_coords),axis = 0)
    # print(nodes.shape)
    
    
    elem_connectivity = np.concat((vtk_connectivity,tet_connectivity,wedge_connectivity),dtype= np.int32)
    cell_types = np.array([HEX.id,HEX.id,TETRA.id,WEDGE.id],np.int32)
    
    
    
    # mesh = Mesh(nodes,elem_connectivity,cell_types)
    # tet_connectivity = np.array([0,0, 1, 2, 3],np.int32)
    # tet_connectivity[0] = 4
    # cell_types = np.array([TETRA.id],np.int32)
    # mesh = Mesh(tet_coords,tet_connectivity,cell_types)
    
    
    wedge_connectivity = np.array([0,0, 1, 2, 3, 4, 5],np.int32)
    wedge_connectivity[0] = 6
    cell_types = np.array([WEDGE.id],np.int32)
    mesh = Mesh(wedge_coords,wedge_connectivity,cell_types)
    
    
    # print(mesh)
    # pv_mesh = to_pyvista(mesh)
    # pv_mesh.plot(show_edges=True)
    
    
    print(mesh.faces.connectivity)
    print(mesh.faces._cell_to_face_array_)
    print(mesh.faces._cell_to_face_offset_)
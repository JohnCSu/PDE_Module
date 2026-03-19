import numpy as np
import numba
from typing import Optional
from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types.cell_types import HEX,TETRA,WEDGE
from pde_module.mesh.face import Faces
from pde_module.mesh.edge import Edges
from pde_module.mesh.topology.topology import Topology
from pde_module.mesh.utils import generate_vectorized_vtk_hex,get_mesh_dimension
from pde_module.mesh.group.group import Group
import pyvista as pv
from dataclasses import dataclass,fields
@dataclass(init=False)
class Mesh:
    """
    Represents a 3D geometry with strongly-typed vertices, cells, and automatically derived edges and faces.
    Supports two modes:
      1. Cells mode: pass a Cells object; edges and faces are derived automatically.
      2. Wireframe mode: pass a raw edges array directly; no cells or face derivation occurs.
    """
    nodes: np.ndarray
    cells: Cells
    faces: Faces
    edges: Edges
    topology: Topology
    dimension:int
    groups:dict[str|Group]
    float_dtype: np.ndarray
    int_dtype: np.ndarray
    def __init__(self, nodes: np.ndarray, cells_connectivity: np.ndarray,cell_types:np.ndarray,dimension:Optional[int]= None,float_dtype = np.float32,int_dtype = np.int32):
        """
        Initializes a Mesh object. Currently implemented for 3D only meshes
        
        Args:
            nodes (np.ndarray): Shape (N, 3), coerced to np.float32.
            cells_connectivity (np.ndarray): Connectivity using nodeIDs flatened. int32 
            cell_types (np.ndarray): 1D array of VTK cell_type IDs see https://docs.pyvista.org/api/utilities/_autosummary/pyvista.celltype for IDs. Only a subset is accepted
            dimension (Optional int): dimension of mesh. if None, the dimension is inferred from cell type otherwise a check is performed to ensure all cell type
                dimensions are less than or equal to the dimension
        """
        # Lets first get the connectivity then calculate everything else like centroids area etc
        self.nodes = np.asarray(nodes, dtype=float_dtype)
        self.cells = Cells(nodes,cells_connectivity,cell_types,float_dtype,int_dtype)
        self.dimension = get_mesh_dimension(self.cells.unique_cell_types,dimension)
        # Calculate Edges. Need this for rendering
        self.edges = Edges.from_cells(self.cells)
        # Calculate All geometric Faces
        self.faces = Faces.from_cells(self.cells)
        
        self.groups = {}
        self.topology = Topology()
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
    
    
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
])
    
    wedge_coords[:,2] += 2.
    
    tet_connectivity = np.array([0,0, 1, 2, 3]) + len(hex_nodes)
    tet_connectivity[0] = 4
    
    wedge_connectivity = np.array([0,0, 1, 2, 3, 4, 5]) + len(hex_nodes) + len(tet_coords)
    wedge_connectivity[0] = 6
    
    nodes = np.concat((hex_nodes,tet_coords,wedge_coords),axis = 0)
    # print(nodes.shape)
    
    
    elem_connectivity = np.concat((vtk_connectivity,tet_connectivity,wedge_connectivity),dtype= np.int32)
    cell_types = np.array([HEX.id,HEX.id,TETRA.id,WEDGE.id],np.int32)
    
    mesh = Mesh(nodes,elem_connectivity,cell_types)
    
    print(mesh)
    # pv_mesh = mesh.to_pyvista()
    # pv_mesh.plot(show_edges=True)
    
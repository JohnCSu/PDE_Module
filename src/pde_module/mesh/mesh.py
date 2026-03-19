import numpy as np
from typing import Optional
from pde_module.mesh.cell import Cells
from pde_module.mesh.face import Faces
from pde_module.mesh.edge import Edges
from pde_module.mesh.topology.topology import Topology
from pde_module.mesh.utils import get_mesh_dimension
from pde_module.mesh.group.group import Group
from dataclasses import dataclass

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
    

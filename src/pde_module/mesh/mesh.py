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
    """Represents a 3D geometry with nodes, cells, faces, and edges.

    Nodes, cells, faces, and edges are stored as strongly-typed arrays.
    Cells and faces are derived automatically from the connectivity.

    Attributes:
        nodes: Array of node coordinates (N, 3).
        cells: Cells object containing connectivity and types.
        faces: Faces object with geometric information.
        edges: Edges object with connectivity.
        topology: Topology object describing connectivity relationships.
        dimension: Dimension of the mesh (1, 2, or 3).
        float_dtype: NumPy float dtype for coordinates.
        int_dtype: NumPy int dtype for indices.
    """

    nodes: np.ndarray
    cells: Cells
    faces: Faces
    edges: Edges
    topology: Topology
    dimension: int
    float_dtype: np.dtype
    int_dtype: np.dtype

    def __init__(
        self,
        nodes: np.ndarray,
        cells_connectivity: np.ndarray,
        cell_types: np.ndarray,
        dimension: Optional[int] = None,
        float_dtype=np.float32,
        int_dtype=np.int32,
    ) -> None:
        """Initialize a Mesh object.

        Args:
            nodes: Shape (N, 3) array of node coordinates.
            cells_connectivity: Connectivity using node IDs (flattened).
            cell_types: 1D array of VTK cell type IDs.
            dimension: Mesh dimension. If None, inferred from cell types.
            float_dtype: NumPy float dtype for coordinates.
            int_dtype: NumPy int dtype for indices.
        """
        self.nodes = np.asarray(nodes, dtype=float_dtype)
        self.cells = Cells(
            nodes, cells_connectivity, cell_types, float_dtype, int_dtype
        )
        self.dimension = get_mesh_dimension(self.cells.unique_cell_types, dimension)
        self.edges = Edges.from_cells(self.cells)
        self.faces = Faces.from_cells(self.cells)

        self.topology = Topology()
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    # def add_group(
    #     self,
    #     name: str,
    #     cell_ids: Optional[np.ndarray] = None,
    #     face_ids: Optional[np.ndarray] = None,
    #     edge_ids: Optional[np.ndarray] = None,
    #     node_ids: Optional[np.ndarray] = None,
    # ) -> None:
    #     """Add a group of IDs to the mesh.

    #     Useful for defining boundary conditions and other subsets.

    #     Args:
    #         name: Name of the group.
    #         cell_ids: Array of cell IDs in the group.
    #         face_ids: Array of face IDs in the group.
    #         edge_ids: Array of edge IDs in the group.
    #         node_ids: Array of node IDs in the group.

    #     Raises:
    #         AssertionError: If name exists or all IDs are None.
    #     """
    #     assert name not in self.groups.keys(), "name for group already exists"
    #     assert any(i is not None for i in [cell_ids, face_ids, edge_ids, node_ids]), (
    #         "Group has no members all are None!"
    #     )
    #     self.groups[name] = Group(
    #         name, cell_ids, face_ids, edge_ids, node_ids, int_dtype=self.int_dtype
    #     )

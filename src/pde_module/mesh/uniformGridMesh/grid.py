import numpy as np
from pde_module.mesh.mesh import Mesh
from math import prod
from pde_module.mesh.uniformGridMesh.functions import *


class UniformGridMesh(Mesh):
    """A uniform grid mesh for structured PDE simulations.

    A structured grid where each cell is a hypercube and nodes are
    uniformly spaced. Supports 1D, 2D, and 3D grids.

    Attributes:
        dx: Grid spacing.
        area: Area of a cell (dx^2).
        volume: Volume of a cell (dx^3).
        ghost_cells: Number of ghost cells.
        nodal_grid: Full nodal coordinate grid.
        dimension: Number of active dimensions.
        origin: Origin point of the grid.
        nodes_per_axis: Number of nodes per axis.
        num_cells: Total number of cells.
        num_nodes: Total number of nodes
        meshgrid: List of coordinate arrays for each dimension.
        coordinate_vectors: Coordinate vectors for each axis.
    """

    dx: float
    area: float
    volume: float
    ghost_cells: int
    nodal_grid: np.ndarray
    dimension: int
    origin: np.ndarray
    nodes_per_axis: tuple[int, ...]
    num_cells: int
    num_nodes: int
    meshgrid: list[np.ndarray]
    coordinate_vectors: tuple[np.ndarray, ...]

    def __init__(
        self,
        dx: float,
        nodes_per_axis: tuple[int, ...],
        origin: np.ndarray | None = None,
        ghost_cells: int | None = None,
        float_dtype=np.float32,
        int_dtype=np.int32,
    ) -> None:
        """Initialize a uniform grid mesh.

        Args:
            dx: Grid spacing.
            nodes_per_axis: Number of nodes per axis as a 3-tuple.
            origin: Origin point. If None, defaults to (0, 0, 0).
            ghost_cells: Number of ghost cells. If None, defaults to 0.
            float_dtype: NumPy float dtype for coordinates.
            int_dtype: NumPy int dtype for indices.
        """
        dx = float_dtype(dx)
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.ghost_cells = get_ghost_cells(ghost_cells)

        self.origin = (
            np.zeros(3, dtype=float_dtype)
            if origin is None
            else np.array(origin, dtype=float_dtype)
        )
        assert len(self.origin) == 3, "Origin must be a tuple of length 3"

        self.nodes_per_axis, self.coordinate_vectors, self.meshgrid, self.nodal_grid = (
            create_nodes_grid(dx, nodes_per_axis, self.origin, self.ghost_cells)
        )

        self.num_cells = prod(i - 1 for i in self.nodes_per_axis if i > 1)
        dimension = sum(1 for i in self.nodes_per_axis if i > 1)
        assert 1 <= dimension <= 3

        cells_connectivity, cell_types = cell_connectivity_and_type(
            self.nodes_per_axis, self.num_cells, dimension, int_dtype
        )
        
        self.num_nodes = prod(self.nodes_per_axis)
        super().__init__(
            self.nodal_grid.reshape(-1, 3),
            cells_connectivity,
            cell_types,
            dimension,
            float_dtype,
            int_dtype,
        )


if __name__ == "__main__":
    import pyvista as pv

    mesh = UniformGridMesh(1, (3, 2, 2), ghost_cells=1)
    pv_mesh = pv.UnstructuredGrid(mesh.cells.connectivity, mesh.cells.types, mesh.nodes)
    pv_mesh.plot(show_edges=True)

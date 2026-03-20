import numpy as np
import numba as nb
from pde_module.mesh.cell_types.cell_types import (
    CellType,
    NUM_NODES_PER_CELL_DICT,
    CELLTYPES_DICT,
)


@nb.njit
def _calculate_cell_centroids(
    nodes: np.ndarray, cells: np.ndarray, cell_offsets: np.ndarray
) -> np.ndarray:
    """Calculate unweighted centroids of each cell.

    Args:
        nodes: Array of node coordinates (N, 3).
        cells: Flattened connectivity array.
        cell_offsets: Starting offset for each cell in connectivity.

    Returns:
        Array of centroids with shape (n_cells, 3).
    """
    n_cells = cell_offsets.shape[0]
    centroids = np.zeros((n_cells, 3), dtype=np.float32)

    for i in range(n_cells):
        start_idx = cell_offsets[i]
        num_nodes = cells[start_idx]
        cell_ids = cells[start_idx + 1 : start_idx + 1 + num_nodes]
        node_locs = nodes[cell_ids, :]
        centroids[i, :] = node_locs.mean(axis=0)
    return centroids


def check_cell_types(
    cell_connectivity: np.ndarray, IDs: np.ndarray, cell_types: np.ndarray
) -> None:
    """Validate that cell types match their connectivity.

    Args:
        cell_connectivity: Flattened connectivity array.
        IDs: Cell offset indices.
        cell_types: VTK cell type IDs.

    Raises:
        ValueError: If a cell's type doesn't match its connectivity.
    """
    id, cell_type, num_nodes = _check_cell_types(
        cell_connectivity, IDs, cell_types, NUM_NODES_PER_CELL_DICT
    )
    if id != -1:
        raise ValueError(
            f"Error with cell type and connectivity. For Cell {id} {num_nodes} nodes were expected "
            f"but got celltype id of {CELLTYPES_DICT[cell_type].id} which is a {CELLTYPES_DICT[cell_type].name} "
            f"cell which has {CELLTYPES_DICT[cell_type].num_nodes} number of nodes"
        )


@nb.njit(cache=True, parallel=False)
def _check_cell_types(
    cell_connectivity: np.ndarray,
    IDs: np.ndarray,
    cell_types: np.ndarray,
    num_nodes_dict,
) -> tuple[int, int, int]:
    """Numba helper to check cell type consistency.

    Returns:
        Tuple of (cell_id, cell_type, num_nodes) if error, or (-1, -1, -1) if OK.
    """
    for id in nb.prange(len(IDs)):
        offset = IDs[id]
        num_nodes = cell_connectivity[offset]

        if num_nodes != num_nodes_dict[cell_types[id]]:
            return id, cell_types[id], num_nodes
    return -1, -1, -1

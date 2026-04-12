import numpy as np
import numba as nb
from pde_module.mesh.cell_types.cell_types import (
    LOCAL_EDGE_ORDERING_DICT,
    CELLTYPES_DICT,
)
from pde_module.mesh.cell import Cells


def get_edges(cells: Cells) -> np.ndarray:
    """Extract unique edges from cells.

    Args:
        cells: Cells object containing connectivity.

    Returns:
        Array of unique edges (E, 2) as node ID pairs.
    """
    max_num_edges = max(
        [CELLTYPES_DICT[key].num_edges for key in cells.unique_cell_types]
    )
    raw_edges = get_raw_edges(
        cells.connectivity,
        cells.IDs,
        cells.types,
        max_num_edges,
        LOCAL_EDGE_ORDERING_DICT,
    )
    edges = np.unique(raw_edges, axis=0)
    if sum(edges[0]) == -2:
        edges = edges[1:]
    return edges


@nb.njit
def get_raw_edges(
    cell_connectivity: np.ndarray,
    cell_offsets: np.ndarray,
    cell_types: np.ndarray,
    num_max_edges: int,
    local_edge_dict: dict,
) -> np.ndarray:
    """Generate raw edge arrays from cell connectivity.

    Args:
        cell_connectivity: Flattened cell connectivity.
        cell_offsets: Cell offset indices.
        cell_types: Cell type IDs.
        num_max_edges: Maximum edges per cell.
        local_edge_dict: Dictionary mapping cell type to edge ordering.

    Returns:
        Raw edges array (N, 2).
    """
    raw_edges = np.full((len(cell_offsets), num_max_edges, 2), -1,dtype = cell_connectivity.dtype)
    for id in nb.prange(len(cell_offsets)):
        offset = cell_offsets[id]
        num_nodes = cell_connectivity[offset]
        nodes = cell_connectivity[offset + 1 : offset + 1 + num_nodes]
        cell_type = cell_types[id]
        local_edge_ordering = local_edge_dict[cell_type]
        for j in range(local_edge_ordering.shape[0]):
            edge = nodes[local_edge_ordering[j]]
            raw_edges[id, j, 0], raw_edges[id, j, 1] = np.min(edge), np.max(edge)

    return raw_edges.reshape(-1, 2)

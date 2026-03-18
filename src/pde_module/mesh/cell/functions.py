import numpy as np
import numba as nb
from pde_module.mesh.cell_types import CellType

@nb.njit
def _calculate_cell_centroids(nodes: np.ndarray, cells: np.ndarray, cell_offsets: np.ndarray) -> np.ndarray:
    """
    Calculates the unweighted centroid of each cell.
    Returns a (C, 3) array of centroids.
    """
    n_cells = cell_offsets.shape[0]
    centroids = np.zeros((n_cells, 3), dtype=np.float32)
    
    for i in range(n_cells):
        start_idx = cell_offsets[i]
        num_nodes = cells[start_idx]
        cell_ids = cells[start_idx+1:start_idx+1+num_nodes]
        node_locs = nodes[cell_ids,:] # nx3 nodes
        centroids[i,:] = node_locs.mean(axis = 0) # 3 array
    return centroids

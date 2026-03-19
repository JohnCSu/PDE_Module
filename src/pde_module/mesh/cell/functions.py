import numpy as np
import numba as nb
from pde_module.mesh.cell_types.cell_types import CellType,NUM_NODES_PER_CELL_DICT,CELLTYPES_DICT

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

def check_cell_types(cell_connectivity,IDs,cell_types) -> None:
    id,cell_type,num_nodes = _check_cell_types(cell_connectivity,IDs,cell_types,NUM_NODES_PER_CELL_DICT)
    if id != -1:
        raise ValueError(f'Error with cell type and connectivity. For Cell {id} {num_nodes} nodes were expected \
                         but got celltype id of {CELLTYPES_DICT[cell_type].id} which is a {CELLTYPES_DICT[cell_type].name} \
                         cell which has {CELLTYPES_DICT[cell_type].num_nodes} number of nodes')
        
    
@nb.njit(cache=True,parallel = False)
def _check_cell_types(cell_connectivity,IDs,cell_types,num_nodes_dict):
    for id in nb.prange(len(IDs)):
        offset = IDs[id]
        num_nodes = cell_connectivity[offset]
        
        if num_nodes != num_nodes_dict[cell_types[id]]:
            return id,cell_types[id],num_nodes
    return -1,-1,-1
        
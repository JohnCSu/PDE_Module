import numpy as np
import numba as nb
from pde_module.mesh.cell_types.cell_types import (
    CellType,
    NUM_NODES_PER_CELL_DICT,
    CELLTYPES_DICT,
)


@nb.njit(cache = True)
def calculate_cell_centroids(
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
    centroids = np.zeros((n_cells, 3), dtype=nodes.dtype)

    for i in nb.prange(n_cells):
        start_idx = cell_offsets[i]
        num_nodes = cells[start_idx]
        node_IDs = cells[start_idx + 1 : start_idx + 1 + num_nodes]
        
        cx = 0.
        cy = 0.
        cz = 0.
        for j in range(num_nodes):
            cx += nodes[node_IDs[j],0]
            cy += nodes[node_IDs[j],1]
            cz += nodes[node_IDs[j],2]
            
        centroids[i, 0] = cx/num_nodes
        centroids[i, 1] = cy/num_nodes 
        centroids[i, 2] = cz/num_nodes  
    return centroids


@nb.njit(cache = True)
def calculate_cell_volumes(cell_centroids,cell_to_face,cell_to_face_offset,face_centroids,face_normals,dimension):
    volumes = np.empty(len(cell_centroids),dtype = cell_centroids.dtype)
    
    for cell_id in nb.prange(volumes.shape[0]):
        offset = cell_to_face_offset[cell_id]
        num_faces = cell_to_face[offset]
        volume = 0.
        centroid = cell_centroids[cell_id]
        for i in range(1,num_faces+1):
            face_id = cell_to_face[offset + i]
            
            face_centroid = face_centroids[face_id]
            face_normal = face_normals[face_id]
            
            r = face_centroid - centroid
            r_dot_f_n = np.dot(r,face_normal)
            r_dot_f_n = np.where(r_dot_f_n < 0.,-1.,1.)*r_dot_f_n
            volume += r_dot_f_n 
        
        volumes[cell_id] = volume/dimension
    return volumes
            
            



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

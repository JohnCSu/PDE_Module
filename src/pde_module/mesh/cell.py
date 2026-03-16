import numpy as np
import numba as nb
from pde_module.mesh.utils import getIDs,check_IDs
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


class Cells:
    """
    Represents volumetric cells following a VTK-style 1D connectivity array.
    """
    connectivity:np.ndarray
    IDs:np.ndarray
    types:np.ndarray
    float_dtype:np.dtype
    int_dtype: np.dtype
    cell_data:dict[int,tuple[int,np.ndarray]]
    def __init__(self,nodes: np.ndarray, cells_arr: np.ndarray,cell_type,float_dtype = np.float32,int_dtype = np.int32):
        """
        Initializes a Cells object and computes volumes and centroids.
        
        Args:
            cells_arr (np.ndarray): 1D array of np.int32 [num_nodes, n1, n2... num_nodes2, n1, n2...]
            cell_IDs (np.ndarray): 1D array of np.int32 containing the starting offset index for each cell in cells_arr.
            cell_type (np.ndarray): 1D array of 
            nodes (np.ndarray): The parent mesh nodes to calculate volumes.
            
        """
        self.connectivity = np.asarray(cells_arr, dtype=int_dtype)
        self.types = np.astype(cell_type,np.int32) 
        self.unique_cell_types,self.cell_type_count = np.unique(self.types,return_counts=True)
        
        self.IDs = getIDs(cells_arr)
        
        
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        
        assert check_IDs(self.connectivity,self.IDs,len(nodes))
        assert self.IDs.shape == self.types.shape
        

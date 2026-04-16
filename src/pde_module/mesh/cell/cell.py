import numpy as np
import numba as nb
from pde_module.mesh.utils import getIDs, check_IDs
from .functions import check_cell_types,calculate_cell_centroids,calculate_cell_volumes
from dataclasses import dataclass
from typing import Optional

class Cells:
    """Represents volumetric cells with VTK-style connectivity.

    Connectivity is stored as a 1D array where each cell is represented
    as: [num_nodes, n1, n2, ..., nk] for each cell.

    Attributes:
        connectivity: Flattened connectivity array.
        IDs: Starting offset index for each cell in connectivity.
        types: VTK cell type ID for each cell.
        unique_cell_types: Array of unique cell type IDs present.
        cell_type_count: Count of cells for each unique type.
        float_dtype: NumPy float dtype.
        int_dtype: NumPy integer dtype.
    """

    connectivity: np.ndarray
    IDs: np.ndarray
    types: np.ndarray
    float_dtype: np.dtype
    int_dtype: np.dtype
    centroids: Optional[np.ndarray] = None
    volumes: Optional[np.ndarray] = None
    def __init__(
        self,
        nodes: np.ndarray,
        cells_arr: np.ndarray,
        cell_type,
        float_dtype=np.float32,
        int_dtype=np.int32,
    ) -> None:
        """Initialize Cells object.

        Args:
            nodes: Array of node coordinates.
            cells_arr: 1D connectivity array.
            cell_type: Array of VTK cell type IDs.
            float_dtype: NumPy float dtype.
            int_dtype: NumPy integer dtype.
        """
        self.connectivity = np.asarray(cells_arr, dtype=int_dtype)
        self.types = np.astype(cell_type, np.int32)
        self.unique_cell_types, self.cell_type_count = np.unique(
            self.types, return_counts=True
        )

        self.IDs = getIDs(self.connectivity)
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

        assert check_IDs(self.connectivity, self.IDs, len(nodes))
        assert self.IDs.shape == self.types.shape
        check_cell_types(self.connectivity, self.IDs, self.types)

    def get_centroids(self,nodes):
        self.centroids = calculate_cell_centroids(nodes,self.connectivity,self.IDs)
    
    def get_volumes(self,cell_to_face,cell_to_face_offset,face_centroids,face_normals):
        self.volumes = calculate_cell_volumes(self.centroids,cell_to_face,cell_to_face_offset,face_centroids,face_normals,dimension = 3)
    
    def get_neighbors(self,cell_to_face,cell_to_face_offset,ownerNeighbor):
        self.neighbors,self.neighbors_offset = calculate_neighbors(cell_to_face,cell_to_face_offset,ownerNeighbor)
    
    def __len__(self) -> int:
        """Return the number of cells."""
        return len(self.IDs)




@nb.njit(parallel= True,cache = True)
def calculate_neighbors(cell_to_face,cell_to_face_offset,ownerNeighbor):
    # we have at most cell_to_face length
    cell_neighbors = np.full_like(cell_to_face,fill_value=-1,dtype = cell_to_face.dtype)
    
    
    cell_neighbors_count = np.empty(len(cell_to_face_offset) +1, dtype = cell_to_face_offset.dtype)
    cell_neighbors_count[0] = 0
    
    num_cells = len(cell_to_face_offset)
    
    ''' 
    Go through all faces and check first if owner or neighbor. 
        If owner, check if boundary or interior face. if boundary add -1 otherwise add neighborID
        If NeighborID then add ownerID as guanteed to be interior face.
        Also track the number of neighbors (<= number of faces in a cell) to total neighbors and cumsum
        to get offsets
        
    Then:
        Create a new array based on total neighbors

    '''
    for cell_id in nb.prange(num_cells):
        offset = cell_to_face_offset[cell_id]
        num_faces = cell_to_face[offset]
        num_neighbors = 0
        # print('hello')
        for j in range(num_faces):
            faceID = cell_to_face[offset + 1 + j]
            ownerID,neighborID = ownerNeighbor[faceID]
            if ownerID == cell_id: # Check the neighbor ID for boundary. If not leave as -1
                is_interior = neighborID != -1
                cell_neighbors[offset + 1 + num_neighbors] = np.where(is_interior,neighborID,-1)
                num_neighbors += np.int64(is_interior) # Increment by one if face is internal face
            else: # Must be neighborID then just add ownerID
                cell_neighbors[offset + 1 + num_neighbors] = ownerID
                num_neighbors += 1
        
        cell_neighbors_count[cell_id+1] = num_neighbors
        cell_neighbors[offset] = num_neighbors
    # print(cell_neighbors_count)
    cell_neighbors_offsets = np.cumsum(cell_neighbors_count)
    
    total_length = cell_neighbors_offsets[-1] + num_cells
    # print(total_length)
    # print(cell_neighbors_offsets)
    cell_neighbors_out = np.empty(shape = total_length,dtype = cell_to_face.dtype)
    
    for cell_id in nb.prange(num_cells):
        original_offset = cell_to_face_offset[cell_id]
        new_offset = cell_neighbors_offsets[cell_id] + cell_id
        num_neighbors = cell_neighbors[original_offset]
        cell_neighbors_out[new_offset] = num_neighbors
        for j in range(num_neighbors):
            cell_neighbors_out[new_offset+j+1] = cell_neighbors[original_offset+j+1]

        cell_neighbors_offsets[cell_id] = new_offset 
    # print(cell_neighbors_offsets)
    
    return cell_neighbors_out,cell_neighbors_offsets[:-1] 
        
        
        
        
        
    
    
    
    
    
                

            
            
            
            
            
        
        
    
    
    
    
        
        
        
    
    
    
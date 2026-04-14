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
    
    def __len__(self) -> int:
        """Return the number of cells."""
        return len(self.IDs)




        
        
        
    
    
    
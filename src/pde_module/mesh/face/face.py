import numpy as np
import numba as nb
from ..cell import Cells
from .functions import get_faces,get_ownerNeighbor,calculate_faces_area_normals_centroids
from typing import Optional

class Faces:
    """Represents faces (polygons) with VTK-style connectivity.

    Attributes:
        connectivity: Flattened face connectivity array.
        IDs: Starting offset index for each face.
        int_dtype: NumPy integer dtype.
        float_dtype: NumPy float dtype.
    """

    connectivity: np.ndarray
    IDs: np.ndarray
    float_dtype: np.dtype
    int_dtype: np.dtype
    _cell_to_face_array_:np.ndarray
    _cell_to_face_offset_:np.ndarray
    
    centroids:Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    ownerNeighbor:Optional[np.ndarray] = None
    '''
    F,2 Array, mapping owner cell to neighbor cell, -1 means no neighbor (i.e boundary)
    '''
    
    '''
    Store the normal vector of each face. To save compute, The normal vector is not normalised so the magnitude
    of the vector is equal to the area of the face
    '''
    
    def __init__(
        self,
        connectivity: np.ndarray,
        IDs: np.ndarray,
        float_dtype=np.float32,
        int_dtype=np.int32,
    ) -> None:
        """Initialize Faces object if face information is already known. Otherwise use the classmethod `from_cells()` to
        initialise from cell data 

        Args:
            connectivity: Flattened face connectivity array.
            IDs: Face offset indices.
            float_dtype: NumPy float dtype.
            int_dtype: NumPy integer dtype.
        """
        self.connectivity, self.IDs = connectivity, IDs
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    def get_centroids_and_normals(self,nodes):
        self.normals,self.centroids = calculate_faces_area_normals_centroids(nodes,self.connectivity,self.IDs)
    
    def get_OwnerNeighbor(self):
        self.ownerNeighbor = get_ownerNeighbor(self._cell_to_face_array_,self._cell_to_face_offset_,len(self))
    
    @classmethod
    def from_cells(cls, cells: Cells) -> "Faces":
        """Create Faces from a Cells object.

        Args:
            cells: The parent Cells object.

        Returns:
            New Faces object.
        """
        assert isinstance(cells, Cells)
        face_connectivity, face_IDs,cell_face_array, cell_face_offsets = get_faces(cells)
        face_obj = cls(face_connectivity, face_IDs, cells.float_dtype, cells.int_dtype)
        
        face_obj._cell_to_face_array_,face_obj._cell_to_face_offset_ = (cell_face_array, cell_face_offsets)
        return face_obj
    
    def __len__(self) -> int:
        """Return the number of faces."""
        return len(self.IDs)



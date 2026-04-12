import numpy as np
import numba as nb
from dataclasses import dataclass


@dataclass
class Flat_array_container:
    """Container for topology arrays.

    Attributes:
        array: The topology array.
        offsets: Optional offset indices for variable-length data.
    """

    array: np.ndarray
    offsets: np.ndarray | None


class Topology:
    """Describes connectivity between mesh entities of different dimensions. For now we have 3D Cells to Faces

    Stores face-to-cell, cell-to-face, and cell-to-cell connectivity.

    Attributes:
        face_to_cell: Connectivity from faces to cells.
        cell_to_face: Connectivity from cells to faces.
        cell_to_cell: Connectivity from cells to neighboring cells.
    """

    face_to_cell: np.ndarray
    cell_to_face: np.ndarray
    cell_to_cell: np.ndarray
    _not_empty: bool
    def __init__(self,cell_to_face_array= None,cell_to_face_offset= None):
        if cell_to_face_array is not None:
            assert cell_to_face_offset is not None
            self.cell_to_face = cell_to_face_array
            self.cell_to_face_offset = cell_to_face_offset
            self._not_empty = True
        else:
            self._not_empty = False
            
    def set_face_to_cell(self,num_faces):
        assert self._not_empty, 'cell_to_face_array must be passed in when initialising Topology'
        self.face_to_cell = get_face_to_cell(self.cell_to_face,self.cell_to_face_offset,num_faces)
        
    def set_cell_to_cell(self):
        if hasattr(self,'face_to_cell'):
            self.cell_to_cell = get_cell_to_cell(self.cell_to_face,self.cell_to_face_offset,self.face_to_cell)
        else:
            raise AttributeError('face_to_cell must be initialised before cell-to-cell can be calculated')
    

@nb.njit(cache=True)
def get_face_to_cell(
    cell_to_face: np.ndarray, cell_to_face_offsets: np.ndarray, num_unique_faces: int
) -> np.ndarray:
    """Build face-to-cell connectivity from cell-to-face data.

    Args:
        cell_to_face: Flattened cell-to-face array.
        cell_to_face_offsets: Offset indices for each cell.
        num_unique_faces: Number of unique faces.

    Returns:
        Array of shape (num_unique_faces, 2) with cell IDs for each face.
    """
    face_to_cell = np.full((num_unique_faces, 2), -1, dtype=cell_to_face.dtype)

    for cell_id in range(len(cell_to_face_offsets)):
        offset = cell_to_face_offsets[cell_id]
        num_faces = cell_to_face[offset]
        for j in range(num_faces):
            faceID = cell_to_face[offset + j + 1]
            idx = np.int32(face_to_cell[faceID, 0] != -1)
            face_to_cell[faceID, idx] = cell_id
    return face_to_cell


@nb.njit(cache=True)
def get_cell_to_cell(
    cell_to_face: np.ndarray, cell_to_face_offsets: np.ndarray, face_to_cell: np.ndarray
) -> np.ndarray:
    """Build cell-to-cell connectivity from cell-to-face data.

    Args:
        cell_to_face: Flattened cell-to-face array.
        cell_to_face_offsets: Offset indices for each cell.
        face_to_cell: Face-to-cell connectivity array.

    Returns:
        Cell-to-cell connectivity array.
    """
    cell_to_cell = np.full_like(cell_to_face, -1)

    for cell_id in range(len(cell_to_face_offsets)):
        offset = cell_to_face_offsets[cell_id]
        num_faces = cell_to_face[offset]
        cell_to_cell[offset] = num_faces

        for j in range(num_faces):
            faceID = cell_to_face[offset + j + 1]
            idx = np.int32(face_to_cell[faceID, 0] == cell_id)
            neighbor_cell_id = face_to_cell[faceID, idx]
            cell_to_cell[offset + j + 1] = neighbor_cell_id

    return cell_to_cell

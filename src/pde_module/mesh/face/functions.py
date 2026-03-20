import numpy as np
import numba as nb
from pde_module.mesh.utils import (
    getIDs,
    check_IDs,
    flatten_and_filter_2D_array,
    sort_rows,
)
from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types.cell_types import (
    LOCALFACEORDERING_DICT,
    CELLTYPES_DICT,
    LOCAL_EDGE_ORDERING_DICT,
)


def get_faces(cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Extract all faces from cells.

    Args:
        cells: Cells object containing connectivity.

    Returns:
        Tuple of (face_connectivity, face_ids).
    """
    max_num_faces = cells.int_dtype(
        max([CELLTYPES_DICT[key].num_faces for key in cells.unique_cell_types])
    )
    max_num_nodes = cells.int_dtype(
        max([CELLTYPES_DICT[key].max_nodes_per_face for key in cells.unique_cell_types])
    )

    cell_face_array = _get_3d_facets_(
        cells.connectivity,
        cells.IDs,
        cells.types,
        max_num_faces,
        max_num_nodes,
        LOCALFACEORDERING_DICT,
    )

    unique_faces, face_ids = get_unique_faces(cell_face_array)
    return unique_faces, face_ids


def get_unique_faces(cell_face_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find unique faces from cell face array.

    Args:
        cell_face_array: Array of faces from all cells.

    Returns:
        Tuple of (unique_faces, face_ids).
    """
    faces_array = cell_face_array.reshape(-1, cell_face_array.shape[-1])
    sorted_faces = sort_rows(faces_array)
    _, unique_indices, unique_inverse = np.unique(
        sorted_faces, axis=0, return_index=True, return_inverse=True
    )
    unique_faces = np.ascontiguousarray(
        faces_array[unique_indices], dtype=cell_face_array.dtype
    )

    if np.sum(unique_faces[0]) == -unique_faces.shape[-1]:
        unique_faces = unique_faces[1:]

    unique_faces, face_ids = flatten_and_filter_2D_array(unique_faces)
    return unique_faces, face_ids


@nb.njit(cache=True)
def _get_2d_facets_(
    cells_connectivity: np.ndarray,
    cellIDs: np.ndarray,
    cellTypes: np.ndarray,
    max_num_faces: int,
    max_num_nodes: int,
    edge_ordering: dict,
) -> np.ndarray:
    """Extract 2D facets (edges) from cells.

    Args:
        cells_connectivity: Flattened cell connectivity.
        cellIDs: Cell offset indices.
        cellTypes: Cell type IDs.
        max_num_faces: Maximum number of faces per cell.
        max_num_nodes: Maximum nodes per face.
        edge_ordering: Dictionary mapping cell type to edge ordering.

    Returns:
        Array of cell faces.
    """
    cell_face_array = np.full(
        (len(cellIDs), max_num_faces, max_num_nodes), -1, dtype=cells_connectivity.dtype
    )
    for i, ID in enumerate(cellIDs):
        num_nodes = cells_connectivity[ID]

        nodes = cells_connectivity[ID + 1 : ID + 1 + num_nodes]
        celltype_id = cellTypes[i]
        local_face_ordering = edge_ordering[celltype_id]
        num_faces = len(local_face_ordering)
        k = 1
        for j in range(num_faces):
            num_nodes = local_face_ordering[k]
            face_order = local_face_ordering[k + 1 : k + 1 + num_nodes]
            face = nodes[face_order]
            cell_face_array[i, j, : len(face_order)] = face
            k += 1 + num_nodes

    return cell_face_array


@nb.njit(cache=True)
def _get_3d_facets_(
    cells_connectivity: np.ndarray,
    cellIDs: np.ndarray,
    cellTypes: np.ndarray,
    max_num_faces: int,
    max_num_nodes: int,
    face_ordering: dict,
) -> np.ndarray:
    """Extract 3D facets (faces) from cells.

    Args:
        cells_connectivity: Flattened cell connectivity.
        cellIDs: Cell offset indices.
        cellTypes: Cell type IDs.
        max_num_faces: Maximum number of faces per cell.
        max_num_nodes: Maximum nodes per face.
        face_ordering: Dictionary mapping cell type to face ordering.

    Returns:
        Array of cell faces.
    """
    cell_face_array = np.full(
        (len(cellIDs), max_num_faces, max_num_nodes), -1, dtype=cells_connectivity.dtype
    )

    for i, ID in enumerate(cellIDs):
        num_nodes = cells_connectivity[ID]

        nodes = cells_connectivity[ID + 1 : ID + 1 + num_nodes]
        celltype_id = cellTypes[i]
        local_face_ordering = face_ordering[celltype_id]
        num_faces = local_face_ordering[0]

        k = 1
        for j in range(num_faces):
            num_nodes = local_face_ordering[k]
            face_order = local_face_ordering[k + 1 : k + 1 + num_nodes]
            face = nodes[face_order]
            cell_face_array[i, j, : len(face_order)] = face
            k += 1 + num_nodes

    return cell_face_array

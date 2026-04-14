import numpy as np
import numba as nb
from pde_module.mesh.utils import (
    flatten_and_filter_2D_array,
    sort_rows,
)
from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types.cell_types import (
    LOCAL_FACE_ORDERING_DICT,
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
    # print('hi')
    cell_face_array = _get_3d_facets_(
        cells.connectivity,
        cells.IDs,
        cells.types,
        max_num_faces,
        max_num_nodes,
        LOCAL_FACE_ORDERING_DICT,
    )

    # print('hey')
    # unique_faces, face_ids = get_unique_faces(cell_face_array)
    unique_faces, face_ids,cell_to_face, cell_to_face_ids = get_unique_faces(cell_face_array)
    return unique_faces, face_ids,cell_to_face, cell_to_face_ids


def get_unique_faces(cell_face_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find unique faces from cell face array.

    Args:
        cell_face_array: Array of faces from all cells.

    Returns:
        Tuple of (unique_faces, face_ids).
    """
    
    
    C,F,N = cell_face_array.shape
    
    # Flatten the first 2 dims of cell face so get a list of all faces (with duplicates)
    faces_array = cell_face_array.reshape(-1, cell_face_array.shape[-1])
    
    # Order each row in ascending order to spot duplicates
    sorted_faces = sort_rows(faces_array)
    
    # Get Unique Faces and Inverse, Use sorting unique to gurantee that empty row( all -1s ) is first row
    # inverse is used to generate the cell_to_face array
    _, unique_indices, unique_inverse = np.unique(
        sorted_faces, axis=0, return_index=True, return_inverse=True
    )
    
    # The First Row could be the empty row i.e all -1s ( if we have diffferent face geoms) 
    remove_first_face = False
    if np.sum(faces_array[unique_indices[0]]) == -faces_array.shape[-1]:
        unique_indices = unique_indices[1:]
        remove_first_face = True
        
    # #The First Row could be the empty row i.e all -1s ( if we have diffferent face geoms) 
    
    # 2D array, Each row is a unique face of nodes and make it contiguous
    unique_faces = np.ascontiguousarray(
        faces_array[unique_indices], dtype=cell_face_array.dtype
    )
    
    unique_faces, face_ids = flatten_and_filter_2D_array(unique_faces)
    cell_to_face, cell_to_face_ids = flatten_and_filter_2D_array(unique_inverse.reshape((C,F)).astype(cell_face_array.dtype),filter_value=unique_indices[0] if remove_first_face else -1) # Ignore the filter
    # print('hello')
    return unique_faces, face_ids,cell_to_face, cell_to_face_ids

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
    # print('ho')
    
    # I think this can be parallised?
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



@nb.njit(cache=True)
def get_ownerNeighbor(
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
    ownerNeighbor = np.full((num_unique_faces, 2), -1, dtype=cell_to_face.dtype)

    for cell_id in range(len(cell_to_face_offsets)):
        offset = cell_to_face_offsets[cell_id]
        num_faces = cell_to_face[offset]
        for j in range(num_faces):
            faceID = cell_to_face[offset + j + 1]
            idx = np.int32(ownerNeighbor[faceID, 0] != -1)
            ownerNeighbor[faceID, idx] = cell_id
    return ownerNeighbor





@nb.njit(parallel = True,cache=True)
def calculate_faces_area_normals_centroids(
    nodes: np.ndarray, faces: np.ndarray, face_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate area, normal, and centroid for each face.

    Triangulates each face from its centroid and accumulates
    area-weighted normals.

    Args:
        nodes: Node coordinates (N, 3).
        faces: Flattened face connectivity.
        face_offsets: Starting offset for each face.

    Returns:
        Tuple of (areas, normals, centroids), each as an array.
    """
    n_faces = face_offsets.shape[0]
    normals = np.zeros((n_faces, 3), dtype=nodes.dtype)
    centroids = np.zeros((n_faces, 3), dtype=nodes.dtype)
    
    for i in nb.prange(n_faces):
        start_idx = face_offsets[i]
        num_nodes = faces[start_idx]

        if num_nodes < 3:
            continue
        
        cx, cy, cz = 0.0, 0.0, 0.0
        for j in range(num_nodes):
            idx = faces[start_idx + 1 + j]
            cx += nodes[idx, 0]
            cy += nodes[idx, 1]
            cz += nodes[idx, 2]
        cx /= num_nodes
        cy /= num_nodes
        cz /= num_nodes

        centroids[i, 0] = cx
        centroids[i, 1] = cy
        centroids[i, 2] = cz
        
        face_nodes = faces[start_idx + 1:start_idx + 1 + num_nodes]
        
        nx, ny, nz = 0.0, 0.0, 0.0
        for j in range(num_nodes):
            n1 = face_nodes[j]
            n2 = face_nodes[(j+1) % num_nodes]
            
            p1 = nodes[n1]
            p2 = nodes[n2]
            
            nx += (p1[1] - p2[1]) * (p1[2] + p2[2])
            ny += (p1[2] - p2[2]) * (p1[0] + p2[0])
            nz += (p1[0] - p2[0]) * (p1[1] + p2[1])
            
        normals[i,0] = nx/2
        normals[i,1] = ny/2
        normals[i,2] = nz/2
        

        
    return normals, centroids

import numpy as np
import numba as nb
from pde_module.mesh.cell_types import CELLTYPES_DICT
from typing import Optional


def get_mesh_dimension(unique_cell_types: np.ndarray, dim: Optional[int] = None) -> int:
    """Get the dimension of the mesh from cell types.

    Args:
        unique_cell_types: Array of unique VTK cell type IDs.
        dim: Optional specified dimension.

    Returns:
        The mesh dimension.

    Raises:
        AssertionError: If dimension constraints are violated.
    """
    cell_dims = [CELLTYPES_DICT[i].dimension for i in unique_cell_types]

    if dim is None:
        return max(cell_dims)
    else:
        assert isinstance(dim, int), "dim must be None or int"
        assert 1 <= dim <= 3, "dim must be between 1 and 3"
        assert all(celltype_dim <= dim for celltype_dim in cell_dims), (
            "All celltypes must be <= to the specified dim"
        )
        return dim


@nb.njit(cache=True)
def getIDs(connectivity: np.ndarray) -> np.ndarray:
    """Extract cell IDs (starting offsets) from connectivity array.

    Args:
        connectivity: Flattened VTK-style connectivity array.

    Returns:
        Array of starting offsets for each cell.
    """
    buffer = np.ones(shape=len(connectivity), dtype=connectivity.dtype) * (-1)
    buffer_len = len(buffer)

    i = 0
    cell_ID = 0

    while i < buffer_len:
        buffer[cell_ID] = i
        cell_ID += 1
        num_nodes = connectivity[i]
        i += num_nodes + 1

    return buffer[0:cell_ID]


@nb.njit(cache=True)
def check_IDs(connectivity: np.ndarray, IDs: np.ndarray, max_nodeID: int) -> bool:
    """Validate that all node IDs in connectivity are within bounds.

    Args:
        connectivity: Flattened connectivity array.
        IDs: Cell offset indices.
        max_nodeID: Maximum valid node ID.

    Returns:
        True if all IDs are valid, False otherwise.
    """
    for offset in IDs:
        num_nodes = connectivity[offset]
        if np.any(connectivity[offset + 1 : offset + num_nodes + 1] >= max_nodeID):
            return False
    return True


@nb.njit(cache=True)
def flatten_and_filter_2D_array(arr_2D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and filter a 2D array to VTK-style format.

    Removes -1 entries and creates a VTK-style connectivity array.

    Args:
        arr_2D: 2D array with potential -1 padding.

    Returns:
        Tuple of (filtered array, row IDs).
    """
    items_per_row = np.empty(len(arr_2D), arr_2D.dtype)
    for i in range(len(arr_2D)):
        num_items = np.sum((arr_2D[i] > -1).astype(np.int8))
        items_per_row[i] = num_items

    tmp_arr = np.concatenate((items_per_row[:, np.newaxis], arr_2D), axis=1)

    flat_arr_2D = np.ascontiguousarray(tmp_arr.ravel()[tmp_arr.ravel() != -1])
    flat_arr_2D_ids = np.empty_like(items_per_row)
    j = 0
    for i in range(len(flat_arr_2D_ids)):
        num_items = flat_arr_2D[j]
        flat_arr_2D_ids[i] = j
        j += num_items + 1

    return flat_arr_2D, flat_arr_2D_ids


@nb.njit(parallel=False, cache=True)
def sort_rows(arr: np.ndarray) -> np.ndarray:
    """Sort each row of an array.

    Args:
        arr: Input array.

    Returns:
        Array with each row sorted.
    """
    sorted_faces = np.empty_like(arr)
    for i in nb.prange(len(arr)):
        sorted_faces[i] = np.sort(arr[i])
    return sorted_faces


def generate_vectorized_vtk_hex(
    nx: int, ny: int, nz: int, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate node coordinates and VTK hex connectivity vectorized.

    Args:
        nx, ny, nz: Number of cells along each axis.
        dx, dy, dz: Grid spacing for each axis.

    Returns:
        Tuple of (nodes array, VTK connectivity array).
    """
    vx, vy, vz = nx + 1, ny + 1, nz + 1
    x = np.arange(vx) * dx
    y = np.arange(vy) * dy
    z = np.arange(vz) * dz

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    nodes = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    K, J, I = np.meshgrid(k, j, i, indexing="ij")
    base_indices = (I + (J * vx) + (K * vx * vy)).flatten()

    offsets = np.array(
        [
            0,
            1,
            1 + vx,
            vx,
            vx * vy,
            1 + vx * vy,
            1 + vx + vx * vy,
            vx + vx * vy,
        ]
    )

    connectivity = base_indices[:, None] + offsets

    num_cells = nx * ny * nz
    cell_types = np.full((num_cells, 1), 8, dtype=np.int32)
    vtk_cells = np.hstack([cell_types, connectivity]).flatten()

    return nodes, vtk_cells


if __name__ == "__main__":
    nodes, vtk_connectivity = generate_vectorized_vtk_hex(2, 2, 2)
    print(f"Total Nodes: {len(nodes)}")
    ids = getIDs(vtk_connectivity)
    assert check_IDs(vtk_connectivity, ids, len(nodes))


def get_mesh_dimension(unique_cell_types: np.ndarray, dim: Optional[int] = None):
    cell_dims = [CELLTYPES_DICT[i].dimension for i in unique_cell_types]

    if dim is None:
        return max(cell_dims)
    else:
        assert isinstance(dim, int), "dim must be None or int"
        assert 1 <= dim <= 3, "dim must be between 1 and 3"
        assert all((celltype_dim <= dim) for celltype_dim in cell_dims), (
            "All celltypes must be <= to the specified dim"
        )
        return dim


@nb.njit(cache=True)
def getIDs(connectivity: np.ndarray):
    buffer = np.full(shape=len(connectivity),fill_value = int(-1), dtype=connectivity.dtype)
    buffer_len = len(buffer)

    i = 0
    cell_ID = 0

    while i < buffer_len:
        buffer[cell_ID] = i
        cell_ID += 1
        num_nodes = connectivity[i]
        i += num_nodes + 1

    return buffer[0:cell_ID]


@nb.njit(cache=True)
def check_IDs(connectivity: np.ndarray, IDs: np.ndarray, max_nodeID: int):
    for offset in IDs:  # Can be parallised
        num_nodes = connectivity[offset]
        if np.any(connectivity[offset + 1 : offset + num_nodes + 1] >= max_nodeID):
            return False
    return True


@nb.njit(cache=True)
def flatten_and_filter_2D_array(arr_2D,filter_value = -1):
    """
    For an arbitary N,M array, remove the targer value (default -1) and flatten to get a vtk style array. Must be integer array
    """
    items_per_row = np.empty(len(arr_2D), arr_2D.dtype)
    for i in range(len(arr_2D)):
        num_items = np.sum((arr_2D[i] != filter_value).astype(np.int8))
        items_per_row[i] = num_items

    tmp_arr = np.concatenate((items_per_row[:, np.newaxis], arr_2D), axis=1)

    flat_arr_2D = np.ascontiguousarray(tmp_arr.ravel()[tmp_arr.ravel() != filter_value])
    # Get IDs
    flat_arr_2D_ids = np.empty_like(items_per_row,dtype=arr_2D.dtype)
    j = 0
    for i in range(len(flat_arr_2D_ids)):
        num_items = flat_arr_2D[j]
        flat_arr_2D_ids[i] = j
        j += num_items + 1

    return flat_arr_2D, flat_arr_2D_ids


@nb.njit(parallel=False, cache=True)
def sort_rows(arr):
    sorted_faces = np.empty_like(arr)
    for i in nb.prange(len(arr)):
        sorted_faces[i] = np.sort(arr[i])
    return sorted_faces


def generate_vectorized_vtk_hex(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    """
    Vectorized generation of node coordinates and VTK hex connectivity.
    """
    # 1. Generate Node Coordinates
    vx, vy, vz = nx + 1, ny + 1, nz + 1
    x = np.arange(vx) * dx
    y = np.arange(vy) * dy
    z = np.arange(vz) * dz

    # Create the coordinate grid
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    nodes = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

    # 2. Vectorized Connectivity
    # Create a base grid of "lower-left-bottom" node indices for each cell
    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    # K, J, I grid of the starting node index for every element
    K, J, I = np.meshgrid(k, j, i, indexing="ij")
    base_indices = (I + (J * vx) + (K * vx * vy)).flatten()

    # Define offsets for the 8 nodes of a VTK_HEXAHEDRON relative to base_index
    # Order: 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0),
    #        4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    offsets = np.array(
        [
            0,  # node 0: (i, j, k)
            1,  # node 1: (i+1, j, k)
            1 + vx,  # node 2: (i+1, j+1, k)
            vx,  # node 3: (i, j+1, k)
            vx * vy,  # node 4: (i, j, k+1)
            1 + vx * vy,  # node 5: (i+1, j, k+1)
            1 + vx + vx * vy,  # node 6: (i+1, j+1, k+1)
            vx + vx * vy,  # node 7: (i, j+1, k+1)
        ]
    )

    # Use broadcasting to add offsets to all base_indices at once
    # base_indices[:, None] is (N_elements, 1), offsets is (8,)
    # Result is (N_elements, 8)
    connectivity = base_indices[:, None] + offsets

    # 3. VTK Formatting (Prepend the '8' for cell node count)
    num_cells = nx * ny * nz
    cell_types = np.full((num_cells, 1), 8, dtype=np.int32)
    vtk_cells = np.hstack([cell_types, connectivity]).flatten()

    return nodes, vtk_cells


if __name__ == "__main__":
    nodes, vtk_connectivity = generate_vectorized_vtk_hex(2, 2, 2)
    print(f"Total Nodes: {len(nodes)}")
    ids = getIDs(vtk_connectivity)
    assert check_IDs(vtk_connectivity, ids, len(nodes))

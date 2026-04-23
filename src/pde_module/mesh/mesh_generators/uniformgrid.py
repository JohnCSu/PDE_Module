import numpy as np
import numba as nb
from pde_module.mesh.cell_types import CELLTYPES_DICT, HEX, QUAD, EDGE
from typing import Optional, Callable
from math import prod




def create_Uniform_grid(
    dx: float,
    nodes_per_axis: tuple[int, ...],
    origin: np.ndarray = (0.,0.,0.),
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], list[np.ndarray], np.ndarray]:
    """Create node coordinate grid for a uniform mesh.

    Args:
        dx: Grid spacing.
        nodes_per_axis: Number of nodes per axis.
        origin: Origin point.
        ghost_cells: Number of ghost cells.

    Returns:
        Tuple of nodes, cell_connectivity and cell_types
    """
    origin = np.array(origin)
    
    nodal_coordinates_vectors = tuple(
        np.arange(0, axis, dtype=origin.dtype) * dx - axis_origin
        for axis, axis_origin in zip(nodes_per_axis, origin)
    )
    
    meshgrid = np.meshgrid(*nodal_coordinates_vectors, indexing="ij")
    grid = np.stack(meshgrid, axis=-1)
    nodes_per_axis = tuple(len(g) for g in nodal_coordinates_vectors)
    
    num_cells = prod(g-1 if g > 1 else g  for g in nodes_per_axis)
    dimension = sum(1 for g in nodes_per_axis if g > 1)
    
    cell_connectivity,cell_types = cell_connectivity_and_type(nodes_per_axis,num_cells,dimension)
    
    return grid.reshape((-1,3)),cell_connectivity,cell_types


def cell_connectivity_and_type(
    nodes_per_axis: tuple[int, ...],
    num_cells: int,
    dimension: int,
    int_dtype=np.int32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cell connectivity and types for a uniform grid.

    Args:
        nodes_per_axis: Number of nodes per axis.
        num_cells: Total number of cells.
        dimension: Grid dimension (1, 2, or 3).
        int_dtype: NumPy integer dtype.

    Returns:
        Tuple of (connectivity array, cell types array).
    """
    cell_types = [EDGE, QUAD, HEX]
    dtype_dummy = np.zeros(0, dtype=int_dtype)

    connectivity = get_connectivity_vtk(*nodes_per_axis, dtype_dummy)

    cell_type = cell_types[dimension - 1]
    cell_type_arr = np.full(num_cells, cell_type.id, dtype=int_dtype)
    return connectivity, cell_type_arr


@nb.njit(parallel=True, cache=True)
def get_connectivity_vtk(nx: int, ny: int, nz: int, arr_dtype) -> np.ndarray:
    """Generate VTK-style connectivity array for a uniform grid.

    Args:
        nx, ny, nz: Number of nodes per axis.
        arr_dtype: Array dtype for the output.

    Returns:
        Flattened connectivity array in VTK format.
    """
    dims = 0
    if nx > 1:
        dims += 1
    if ny > 1:
        dims += 1
    if nz > 1:
        dims += 1

    if dims == 0:
        return np.zeros(0, dtype=arr_dtype.dtype)

    num_nodes_per_cell = 2**dims
    row_width = num_nodes_per_cell + 1

    cx = max(1, nx - 1)
    cy = max(1, ny - 1)
    cz = max(1, nz - 1)
    num_cells = cx * cy * cz

    connectivity = np.empty((num_cells, row_width), dtype=arr_dtype.dtype)

    if dims == 3:
        slice_size = ny * nz
        for i in nb.prange(cx):
            i_offset = i * slice_size
            i_next_offset = (i + 1) * slice_size

            for j in range(cy):
                j_offset = j * nz
                j_next_offset = (j + 1) * nz

                base_idx_0 = i_offset + j_offset
                base_idx_1 = i_next_offset + j_offset
                base_idx_2 = i_next_offset + j_next_offset
                base_idx_3 = i_offset + j_next_offset

                for k in range(cz):
                    cell_idx = i * (cy * cz) + j * cz + k

                    connectivity[cell_idx, 0] = 8
                    connectivity[cell_idx, 1] = base_idx_0 + k
                    connectivity[cell_idx, 2] = base_idx_1 + k
                    connectivity[cell_idx, 3] = base_idx_2 + k
                    connectivity[cell_idx, 4] = base_idx_3 + k
                    connectivity[cell_idx, 5] = base_idx_0 + k + 1
                    connectivity[cell_idx, 6] = base_idx_1 + k + 1
                    connectivity[cell_idx, 7] = base_idx_2 + k + 1
                    connectivity[cell_idx, 8] = base_idx_3 + k + 1

    elif dims == 2:
        stride_a = ny * nz if nx > 1 else nz
        stride_b = nz if ny > 1 else 1
        ca, cb = (cx, cy) if nx > 1 and ny > 1 else ((cx, cz) if nx > 1 else (cy, cz))

        for i in nb.prange(ca):
            for j in range(cb):
                idx = i * cb + j
                offset = i * stride_a + j * stride_b

                connectivity[idx, 0] = 4
                connectivity[idx, 1] = offset
                connectivity[idx, 2] = offset + stride_a
                connectivity[idx, 3] = offset + stride_a + stride_b
                connectivity[idx, 4] = offset + stride_b

    elif dims == 1:
        stride = 1
        if nx > 1:
            stride = ny * nz
        elif ny > 1:
            stride = nz

        count = max(cx, max(cy, cz))
        for i in nb.prange(count):
            connectivity[i, 0] = 2
            connectivity[i, 1] = i * stride
            connectivity[i, 2] = (i + 1) * stride

    return connectivity.ravel()

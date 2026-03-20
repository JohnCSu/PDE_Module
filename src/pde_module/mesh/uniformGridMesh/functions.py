import numpy as np
import numba as nb
from pde_module.mesh.cell_types import CELLTYPES_DICT, HEX, QUAD, EDGE
from typing import Optional, Callable


def get_ghost_cells(ghost_cells: int | None) -> int:
    """Validate and return ghost cells value.

    Args:
        ghost_cells: Number of ghost cells or None.

    Returns:
        Number of ghost cells (0 if None).

    Raises:
        AssertionError: If ghost_cells is not a non-negative integer.
    """
    ghost_cells = 0 if ghost_cells is None else ghost_cells
    assert isinstance(ghost_cells, int)
    assert ghost_cells >= 0
    return ghost_cells


def add_ghost_coords(
    dx: float, coord_vectors: tuple[np.ndarray, ...], ghost_cells: int
) -> list[np.ndarray]:
    """Add ghost cell coordinates on both sides of each axis.

    Args:
        dx: Grid spacing.
        coord_vectors: Tuple of coordinate vectors for each axis.
        ghost_cells: Number of ghost cells to add on each side.

    Returns:
        List of coordinate vectors including ghost cells.
    """
    if ghost_cells == 0:
        return list(coord_vectors)

    coord_with_ghost = []
    for coord_vector in coord_vectors:
        if len(coord_vector) == 1:
            coord_with_ghost.append(coord_vector)
        else:
            left_g = np.array(
                [coord_vector[0] - n * dx for n in range(1, ghost_cells + 1)]
            )
            right_g = np.array(
                [coord_vector[-1] + n * dx for n in range(1, ghost_cells + 1)]
            )
            coord_with_ghost.append(
                np.concatenate(
                    (left_g, coord_vector, right_g), dtype=coord_vector.dtype
                )
            )
    return coord_with_ghost


def create_nodes_grid(
    dx: float,
    num_points: tuple[int, ...],
    origin: np.ndarray,
    ghost_cells: int,
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], list[np.ndarray], np.ndarray]:
    """Create node coordinate grid for a uniform mesh.

    Args:
        dx: Grid spacing.
        num_points: Number of nodes per axis.
        origin: Origin point.
        ghost_cells: Number of ghost cells.

    Returns:
        Tuple of:
            - nodes_per_axis: Number of nodes per axis including ghost cells.
            - coordinate_vectors: Coordinate vectors for each axis.
            - meshgrid: List of coordinate arrays for each dimension.
            - nodal_grid: Full 3D coordinate grid.
    """
    nodal_coordinates_vectors = tuple(
        np.arange(0, axis, dtype=origin.dtype) * dx - axis_origin
        for axis, axis_origin in zip(num_points, origin)
    )
    ghost_coord_vectors = add_ghost_coords(dx, nodal_coordinates_vectors, ghost_cells)
    meshgrid = np.meshgrid(*ghost_coord_vectors, indexing="ij")
    grid = np.stack(meshgrid, axis=-1)
    nodes_per_axis = tuple(len(g) for g in ghost_coord_vectors)
    meshgrid = [grid[:, :, :, i] for i in range(3)]
    return nodes_per_axis, ghost_coord_vectors, meshgrid, grid


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

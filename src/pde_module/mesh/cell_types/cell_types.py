"""Cell type definitions for VTK-style meshes.

Defines geometric properties of various cell types including
nodes, faces, edges, and their local orderings.
"""

import numpy as np
from dataclasses import dataclass
from numba.typed import Dict
import numba as nb


@dataclass(frozen=True)
class CellType:
    """Describes a VTK cell type with its geometric properties.

    Attributes:
        name: Human-readable name of the cell type.
        id: VTK cell type identifier.
        dimension: Topological dimension (0-3).
        num_nodes: Number of nodes per cell.
        num_faces: Number of faces per cell.
        max_nodes_per_face: Maximum nodes in any face.
        num_edges: Number of edges per cell.
        edges: Array of local edge node indices.
        faces: Flattened array of face node orderings.
    """

    name: str
    id: int
    dimension: int
    num_nodes: int
    num_faces: int
    max_nodes_per_face: int
    num_edges: int
    edges: np.ndarray
    faces: np.ndarray


hex_faces_flat = np.array(
    [
        6,
        4,
        0,
        4,
        7,
        3,
        4,
        1,
        2,
        6,
        5,
        4,
        0,
        1,
        5,
        4,
        4,
        2,
        3,
        7,
        6,
        4,
        0,
        3,
        2,
        1,
        4,
        4,
        5,
        6,
        7,
    ],
    dtype=np.int32,
)

hex_edges_local = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)

tet_faces_flat = np.array(
    [
        4,
        3,
        0,
        1,
        3,
        3,
        1,
        2,
        3,
        3,
        2,
        0,
        3,
        3,
        0,
        2,
        1,
    ],
    dtype=np.int32,
)

tet_edges_local = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        [2, 3],
    ],
    dtype=np.int32,
)

wedge_faces_flat = np.array(
    [
        5,
        3,
        0,
        1,
        2,
        3,
        3,
        5,
        4,
        4,
        0,
        3,
        4,
        1,
        4,
        1,
        4,
        5,
        2,
        4,
        2,
        5,
        3,
        0,
    ],
    dtype=np.int32,
)

wedge_edges_local = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [0, 3],
        [1, 4],
        [2, 5],
    ],
    dtype=np.int32,
)

quad_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32)
tri_edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)

VERTEX = CellType(
    "VERTEX",
    1,
    0,
    1,
    0,
    0,
    0,
    np.zeros((0, 0), dtype=np.int32),
    np.array([0, 0], dtype=np.int32),
)
EDGE = CellType(
    "EDGE",
    3,
    1,
    2,
    0,
    0,
    1,
    np.array([[0, 1]], dtype=np.int32),
    np.array([0, 0], dtype=np.int32),
)
TRIANGLE = CellType(
    "TRIANGLE",
    5,
    2,
    3,
    1,
    3,
    3,
    edges=tri_edges,
    faces=np.array([1, 3, 0, 1, 2], np.int32),
)
QUAD = CellType(
    "QUAD",
    9,
    2,
    4,
    1,
    4,
    4,
    edges=quad_edges,
    faces=np.array([1, 4, 0, 1, 2, 3], np.int32),
)
TETRA = CellType(
    "TETRA", 10, 3, 4, 4, 3, 6, edges=tet_edges_local, faces=tet_faces_flat
)
HEX = CellType("HEX", 12, 3, 8, 6, 4, 12, edges=hex_edges_local, faces=hex_faces_flat)
WEDGE = CellType(
    "WEDGE", 13, 3, 6, 5, 4, 4, edges=wedge_edges_local, faces=wedge_faces_flat
)

CELLTYPES_DICT = {
    5: TRIANGLE,
    9: QUAD,
    10: TETRA,
    12: HEX,
    13: WEDGE,
}

LOCALFACEORDERING_DICT = Dict.empty(
    key_type=nb.types.int32,
    value_type=nb.types.int32[:],
)

NUM_NODES_PER_CELL_DICT = Dict.empty(
    key_type=nb.types.int32,
    value_type=nb.types.int32,
)

LOCAL_EDGE_ORDERING_DICT = Dict.empty(
    key_type=nb.types.int32,
    value_type=nb.types.int32[:, :],
)

for key in CELLTYPES_DICT.keys():
    LOCALFACEORDERING_DICT[np.int32(key)] = CELLTYPES_DICT[key].faces
    LOCAL_EDGE_ORDERING_DICT[np.int32(key)] = CELLTYPES_DICT[key].edges
    NUM_NODES_PER_CELL_DICT[np.int32(key)] = np.int32(CELLTYPES_DICT[key].num_nodes)

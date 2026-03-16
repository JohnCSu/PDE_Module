# TETRA = 10
# HEX = 12
# WEDGE = 13
import numpy as np
from dataclasses import dataclass
from numba.typed import Dict
import numba as nb

@dataclass(frozen=True) 
class CellType:
    id:int
    num_nodes:int
    num_faces:int
    max_nodes_per_face:int
    num_edges:int
    edges:np.ndarray
    faces: np.ndarray
    
# Total faces: 6
hex_faces_flat = np.array([
    6, # Num faces
    4, 0, 4, 7, 3,  # Face 0: Left
    4, 1, 2, 6, 5,  # Face 1: Right
    4, 0, 1, 5, 4,  # Face 2: Front
    4, 2, 3, 7, 6,  # Face 3: Back
    4, 0, 3, 2, 1,  # Face 4: Bottom
    4, 4, 5, 6, 7   # Face 5: Top
], dtype=np.int32)

hex_edges_local = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
],dtype=np.int32)

tet_faces_flat = np.array([
    4, # Num faces
    3, 0, 1, 3,  # Face 0: 3 nodes, indices [0, 1, 3]
    3, 1, 2, 3,  # Face 1: 3 nodes, indices [1, 2, 3]
    3, 2, 0, 3,  # Face 2: 3 nodes, indices [2, 0, 3]
    3, 0, 2, 1   # Face 3: 3 nodes, indices [0, 2, 1]
], dtype=np.int32)

tet_edges_local = np.array([
    [0, 1], [1, 2], [2, 0], # Base edges
    [0, 3], [1, 3], [2, 3]  # Edges to apex
],dtype=np.int32)

wedge_faces_flat = np.array([
    5, # Num faces
    3, 0, 1, 2,     # Face 0: Triangle (Bottom)
    3, 3, 5, 4,     # Face 1: Triangle (Top)
    4, 0, 3, 4, 1,  # Face 2: Quad (Side)
    4, 1, 4, 5, 2,  # Face 3: Quad (Side)
    4, 2, 5, 3, 0   # Face 4: Quad (Side)
], dtype=np.int32)

wedge_edges_local = np.array([
    [0, 1], [1, 2], [2, 0],
    [3, 4], [4, 5], [5, 3],
    [0, 3], [1, 4], [2, 5]
],dtype=np.int32)

TETRA = CellType(10,4,4,3,6,edges = tet_edges_local,faces = tet_faces_flat)
HEX = CellType(12,8,6,4,12,edges = hex_edges_local,faces = hex_faces_flat)
WEDGE = CellType(13,6,5,4,4,edges = wedge_edges_local,faces = wedge_faces_flat)


CELLTYPES_DICT = {
    10:TETRA,
    12:HEX,
    13:WEDGE,
}

LOCALFACEORDERING_DICT = Dict.empty(
    key_type= nb.types.int32,
    value_type = nb.types.int32[:]
)
for key in CELLTYPES_DICT.keys():
    LOCALFACEORDERING_DICT[np.int32(key)] = CELLTYPES_DICT[key].faces 
    
    

def count_nodeIDs(face):
    num_faces = face[0]
    count = 0
    j = 1
    for i in range(num_faces):
        num_nodes = face[j]
        count += num_nodes
        j += num_nodes+1
    return count


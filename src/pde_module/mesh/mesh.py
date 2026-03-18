import numpy as np
import numba

from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types import HEX,TETRA,WEDGE
from pde_module.mesh.face import Faces,get_faces
from pde_module.mesh.edge import Edges,get_edges
from pde_module.mesh.topology import Topology,get_cell_to_cell,get_face_to_cell
from pde_module.mesh.utils import generate_vectorized_vtk_hex
@numba.njit
def _derive_boundaries_numba(cells: np.ndarray, cell_offsets: np.ndarray):
    """
    Parses a Cells array to uniquely extract all edges and faces.
    Returns:
        edges (E, 2) np.int32
        faces_arr (1D VTK format) np.int32
        face_offsets (F,) np.int32
    """
    n_cells = cell_offsets.shape[0]
    
    # We will accumulate faces and edges into dynamic lists, then convert to arrays and unique them
    # Numba typed lists are faster and less prone to tuple typing issues.
    faces_out_list = []
    edges_out_list = []
    
    for i in range(n_cells):
        start_idx = cell_offsets[i]
        num_nodes = cells[start_idx]
        
        # Tetrahedron (4 nodes -> 4 faces, 6 edges)
        if num_nodes == 4:
            n0 = cells[start_idx + 1]
            n1 = cells[start_idx + 2]
            n2 = cells[start_idx + 3]
            n3 = cells[start_idx + 4]
            
            p = np.int32(-1)
            local_faces = [
                (n0, n1, n3, p),
                (n1, n2, n3, p),
                (n2, n0, n3, p),
                (n0, n2, n1, p)
            ]
            local_edges = [
                (n0, n1), (n1, n2), (n2, n0),
                (n0, n3), (n1, n3), (n2, n3)
            ]
            
        # Prism / Wedge (6 nodes -> 5 faces (2 tris, 3 quads), 9 edges)
        elif num_nodes == 6:
            n0 = cells[start_idx + 1]
            n1 = cells[start_idx + 2]
            n2 = cells[start_idx + 3]
            n3 = cells[start_idx + 4]
            n4 = cells[start_idx + 5]
            n5 = cells[start_idx + 6]
            
            p = np.int32(-1)
            local_faces = [
                (n0, n2, n1, p),       # Bottom tri
                (n3, n4, n5, p),       # Top tri
                (n0, n1, n4, n3),   # Side quad
                (n1, n2, n5, n4),   # Side quad
                (n2, n0, n3, n5)    # Side quad
            ]
            local_edges = [
                (n0, n1), (n1, n2), (n2, n0), # Bottom
                (n3, n4), (n4, n5), (n5, n3), # Top
                (n0, n3), (n1, n4), (n2, n5)  # Pillars
            ]
            
        # Hexahedron (8 nodes -> 6 faces (all quads), 12 edges)
        elif num_nodes == 8:
            n0 = cells[start_idx + 1]
            n1 = cells[start_idx + 2]
            n2 = cells[start_idx + 3]
            n3 = cells[start_idx + 4]
            n4 = cells[start_idx + 5]
            n5 = cells[start_idx + 6]
            n6 = cells[start_idx + 7]
            n7 = cells[start_idx + 8]
            
            local_faces = [
                (n0, n4, n7, n3),
                (n1, n2, n6, n5),
                (n0, n1, n5, n4),
                (n2, n3, n7, n6),
                (n0, n3, n2, n1),
                (n4, n5, n6, n7)
            ]
            local_edges = [
                (n0, n1), (n1, n2), (n2, n3), (n3, n0), # Bottom
                (n4, n5), (n5, n6), (n6, n7), (n7, n4), # Top
                (n0, n4), (n1, n5), (n2, n6), (n3, n7)  # Pillars
            ]
        else:
            continue
            
        # Append all to raw lists
        for e in local_edges:
            edges_out_list.append(e)
                
        for f in local_faces:
            faces_out_list.append(f)
            
    # --- Deduplicate Edges ---
    # We create an array of edges where [smaller, larger] 
    # and then dedupe by sorting.
    raw_edges = np.zeros((len(edges_out_list), 2), dtype=np.int32)
    for i in range(len(edges_out_list)):
        e = edges_out_list[i]
        if e[0] < e[1]:
            raw_edges[i, 0] = e[0]
            raw_edges[i, 1] = e[1]
        else:
            raw_edges[i, 0] = e[1]
            raw_edges[i, 1] = e[0]
            
    # For Numba compatibility we just iterate and keep unique
    # We sort by creating a packed 64 bit int (assumes node IDs < 2^31)
    packed_edges = np.zeros(len(edges_out_list), dtype=np.int64)
    for i in range(len(edges_out_list)):
        packed_edges[i] = (np.int64(raw_edges[i, 0]) << 32) | np.int64(raw_edges[i, 1])
        
    packed_edges.sort()
    
    unique_edges_count = 0
    if len(packed_edges) > 0:
        unique_edges_count = 1
        for i in range(1, len(packed_edges)):
            if packed_edges[i] != packed_edges[i-1]:
                unique_edges_count += 1
                
    out_edges = np.zeros((unique_edges_count, 2), dtype=np.int32)
    if unique_edges_count > 0:
        out_edges[0, 0] = np.int32(packed_edges[0] >> 32)
        out_edges[0, 1] = np.int32(packed_edges[0] & 0xFFFFFFFF)
        curr = 1
        for i in range(1, len(packed_edges)):
            if packed_edges[i] != packed_edges[i-1]:
                out_edges[curr, 0] = np.int32(packed_edges[i] >> 32)
                out_edges[curr, 1] = np.int32(packed_edges[i] & 0xFFFFFFFF)
                curr += 1
            
    # --- Deduplicate Faces ---
    # We do a similar trick for faces. We sort the face nodes to create a unique signature.
    # We'll pack up to 4 nodes into a tuple (since we only have tris and quads).
    seen_face_signatures = []
    unique_faces = []
    
    for f in faces_out_list:
        # Instead of `list(f)`, we can't do that efficiently in Numba.
        # Numba arrays and simple manual sorting is safer for types.
        s0, s1, s2, s3 = f[0], f[1], f[2], f[3]
        
        # Micro sort network for 4 items
        if s0 > s1: s0, s1 = s1, s0
        if s2 > s3: s2, s3 = s3, s2
        if s0 > s2: s0, s2 = s2, s0
        if s1 > s3: s1, s3 = s3, s1
        if s1 > s2: s1, s2 = s2, s1
        
        # s is always length 4 because we pad tris with np.int32(-1).
        # -1 will be sorted to the front: [-1, A, B, C]
        sig = (s0, s1, s2, s3)
            
        # Linear search for now (faces are usually bounded)
        found = False
        for ex_sig in seen_face_signatures:
            if ex_sig == sig:
                found = True
                break
                
        if not found:
            seen_face_signatures.append(sig)
            # Add to unique_faces, keeping the exact 4-tuple format so Numba lists unify
            unique_faces.append(f)
                
    # Reconstruct the VTK faces array from unique faces
    total_face_items = 0
    p = np.int32(-1)
    for f in unique_faces:
        if f[3] == p:
            total_face_items += 1 + 3
        else:
            total_face_items += 1 + 4
        
    out_faces_arr = np.zeros(total_face_items, dtype=np.int32)
    out_face_offsets = np.zeros(len(unique_faces), dtype=np.int32)
    
    # Fill Faces
    idx = 0
    for i in range(len(unique_faces)):
        f = unique_faces[i]
        out_face_offsets[i] = idx
        if f[3] == p:
            out_faces_arr[idx] = 3
            idx += 1
            out_faces_arr[idx] = f[0]
            idx += 1
            out_faces_arr[idx] = f[1]
            idx += 1
            out_faces_arr[idx] = f[2]
            idx += 1
        else:
            out_faces_arr[idx] = 4
            idx += 1
            out_faces_arr[idx] = f[0]
            idx += 1
            out_faces_arr[idx] = f[1]
            idx += 1
            out_faces_arr[idx] = f[2]
            idx += 1
            out_faces_arr[idx] = f[3]
            idx += 1
            
    return out_edges, out_faces_arr, out_face_offsets

@numba.njit
def _check_cells_valid_numba(connectivity: np.ndarray, cell_offsets: np.ndarray, nodes_set: numba.typed.Dict) -> bool:
    """
    Validates a VTK-style connectivity array using the starting offsets.
    connectivity: [numberOfNodes, n1, n2...]
    cell_offsets: starting index of each cell in connectivity array
    """
    if cell_offsets.shape[0] == 0:
        return True
        
    for i in range(cell_offsets.shape[0]):
        start_idx = cell_offsets[i]
        if start_idx >= connectivity.shape[0]:
            return False
        num_nodes_in_cell = connectivity[start_idx]
        if start_idx + 1 + num_nodes_in_cell > connectivity.shape[0]:
            return False
        for j in range(num_nodes_in_cell):
            node_id = connectivity[start_idx + 1 + j]
            if node_id not in nodes_set:
                return False
    return True

class Mesh:
    """
    Represents a 3D geometry with strongly-typed vertices, cells, and automatically derived edges and faces.
    Supports two modes:
      1. Cells mode: pass a Cells object; edges and faces are derived automatically.
      2. Wireframe mode: pass a raw edges array directly; no cells or face derivation occurs.
    """
    nodes: np.ndarray
    cells: Cells
    faces: Faces
    edges: Edges
    topology: Topology = Topology()
    def __init__(self, nodes: np.ndarray, cells_connectivity: np.ndarray,cell_types:np.ndarray,float_dtype = np.float32,int_dtype = np.int32):
        """
        Initializes a Mesh object.
        
        Args:
            nodes (np.ndarray): Shape (N, 3), coerced to np.float32.
            cells (np.ndarray): Connectivity using nodeIDs flatened. int32 
        
        Raises:
            ValueError: If nodes have wrong shape, or cell references a non-existent node.
        """
        # Lets first get the connectivity then calculate everything else like centroids area etc
        self.nodes = np.asarray(nodes, dtype=float_dtype)
        self.cells = Cells(nodes,cells_connectivity,cell_types,float_dtype,int_dtype)
        
        # Calculate Edges. Need this for rendering
        edges = get_edges(self.cells)
        self.edges = Edges(edges,float_dtype,int_dtype)
        
        # Calculate Faces 
        (face_connectivity,face_IDs),(cell_to_face,cell_to_face_ids) = get_faces(self.cells)
        self.faces = Faces(face_connectivity,face_IDs,float_dtype,int_dtype)
        
        # Topology arrays
        face_to_cell = get_face_to_cell(cell_to_face,cell_to_face_ids,len(face_IDs))
        cell_to_cell = get_cell_to_cell(cell_to_face,cell_to_face_ids,face_to_cell)
        
        self.topology.add('cell_to_face',cell_to_face,cell_to_face_ids)
        self.topology.add('cell_to_cell',cell_to_cell)
        self.topology.add('face_to_cell',cell_to_cell)

if __name__ == '__main__':
    hex_nodes, vtk_connectivity = generate_vectorized_vtk_hex(1,1,2)
    
    tet_coords = np.array([
    [0.0, 0.0, 0.0], # Node 0: Origin
    [1.0, 0.0, 0.0], # Node 1: X-axis
    [0.0, 1.0, 0.0], # Node 2: Y-axis
    [0.0, 0.0, 1.0]  # Node 3: Apex (Z-axis)
    ])
    wedge_coords = np.array([
    # Bottom Triangle (z=0)
    [0.0, 0.0, 0.0], # Node 0
    [1.0, 0.0, 0.0], # Node 1
    [0.0, 1.0, 0.0], # Node 2
    # Top Triangle (z=1)
    [0.0, 0.0, 1.0], # Node 3
    [1.0, 0.0, 1.0], # Node 4
    [0.0, 1.0, 1.0]  # Node 5
])
    
    
    tet_connectivity = np.array([0,0, 1, 2, 3]) + len(hex_nodes)
    tet_connectivity[0] = 3
    
    wedge_connectivity = np.array([0,0, 1, 2, 3, 4, 5]) + len(hex_nodes) + len(tet_coords)
    wedge_connectivity[0] = 6
    
    nodes = np.concat((hex_nodes,tet_coords,wedge_coords),axis = 0)
    
    elem_connectivity = np.concat((vtk_connectivity,tet_connectivity,wedge_connectivity),dtype= np.int32)
    cell_types = np.array([HEX.id,HEX.id,TETRA.id,WEDGE.id],np.int32)
    # mesh = Mesh(nodes,elem_connectivity,cell_types)
    #Only Hex
    mesh = Mesh(hex_nodes,vtk_connectivity,cell_types[:2])
    
    print(mesh.edges.connectivity)
    
    # print(mesh.faces.faces)
    # print(mesh.faces.faces.shape)
    # print(mesh.faces.check)
    
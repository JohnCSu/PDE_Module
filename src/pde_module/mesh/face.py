import numpy as np
import numba as nb

from pde_module.mesh.utils import getIDs,check_IDs,flatten_and_filter_2D_array,sort_rows
from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types import LOCALFACEORDERING_DICT,CELLTYPES_DICT

class Faces:
    """
    Represents faces (polygons) following a VTK-style 1D connectivity array.
    """
    faces:np.ndarray
    IDs:np.ndarray
    def __init__(self, connectivity,IDs,float_dtype = np.float32,int_dtype = np.int32):
        """
        Initializes a Faces object and computes normals, area, and centroids.
        
        Args:
            faces_arr (np.ndarray): 1D array of np.int32 [num_nodes, n1, n2...]
            face_IDs (np.ndarray): 1D array of np.int32 offsets.
            nodes (np.ndarray): The parent mesh nodes to calculate faces.
        """
        # if face_arr is not None:
        #     self.faces = np.asarray(face_arr, dtype=int_dtype)
        #     self.IDs = getIDs(face_arr)
        #     assert check_IDs(face_arr,self.IDs,len(nodes))    
        # else:
        (self.connectivity,self.IDs) =  connectivity,IDs 
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype




def get_faces(cells:Cells):
    '''
    We need to get calculate all the faces and nodes from celltype
    '''
    max_num_faces = max([CELLTYPES_DICT[key].num_faces for key in cells.unique_cell_types])
    max_num_nodes = max([CELLTYPES_DICT[key].max_nodes_per_face for key in cells.unique_cell_types])
    cell_face_array = _get_faces_(cells.connectivity,cells.IDs,cells.types,max_num_faces,max_num_nodes,LOCALFACEORDERING_DICT)
    (unique_faces,face_ids),(cell_to_face,cell_to_face_ids) = get_unique_faces(cell_face_array)
    return (unique_faces,face_ids),(cell_to_face,cell_to_face_ids)

@nb.njit(cache = True)
def _get_faces_(cells_connectivity,cellIDs,cellTypes,max_num_faces,max_num_nodes,face_ordering:dict):
    # First chec
    cell_face_array = np.full((len(cellIDs),max_num_faces,max_num_nodes),-1,dtype=cells_connectivity.dtype)
    
    for i,ID in enumerate(cellIDs):
        num_nodes = cells_connectivity[ID]
        
        nodes = cells_connectivity[ID+1:ID+1+num_nodes]
        celltype_id = cellTypes[i]
        # print(celltype_id)
        local_face_ordering = face_ordering[celltype_id]
        num_faces = local_face_ordering[0]
        
        k = 1
        for j in range(num_faces):
            num_nodes= local_face_ordering[k]
            face_order =  local_face_ordering[k+1:k+1+num_nodes]
            face = nodes[face_order]        
            cell_face_array[i,j,:len(face_order)] = face
            k += 1 + num_nodes
                    
    return cell_face_array

        
def get_unique_faces(cell_face_array):
    # Flatten to (C*K,F)
    faces_array = cell_face_array.reshape(-1,cell_face_array.shape[-1])
    #Sort Arrays and then find the unique faces 
    sorted_faces = sort_rows(faces_array)
    # Dont actually care about the unique sorted faces just the indices and inverse
    _,unique_indices,unique_inverse =  np.unique(sorted_faces,axis = 0,return_index= True,return_inverse= True)
    unique_faces = np.ascontiguousarray(faces_array[unique_indices]) 
    cell_to_face = unique_inverse.reshape(cell_face_array.shape[0:2])
    
    if np.sum(unique_faces[0]) == -unique_faces.shape[-1]:# Remove the -1,-1,-1,-1... row if we have multiple cell 
        unique_faces = unique_faces[1:]
        cell_to_face -= 1 #
         
    print(cell_to_face)
    cell_to_face,cell_to_face_ids = flatten_and_filter_2D_array(cell_to_face)
    unique_faces,face_ids = flatten_and_filter_2D_array(unique_faces)
    
    return (unique_faces,face_ids),(cell_to_face,cell_to_face_ids)
    


@nb.njit(cache = True)
def _calculate_faces_area_normals_centroids(nodes: np.ndarray, faces: np.ndarray, face_offsets: np.ndarray):
    """
    Calculates the unit normal, area, and unweighted centroid for each face by triangulating from the centroid.
    Returns (areas, normals, centroids).
    """
    n_faces = face_offsets.shape[0]
    areas = np.zeros(n_faces, dtype=np.float32)
    normals = np.zeros((n_faces, 3), dtype=np.float32)
    centroids = np.zeros((n_faces, 3), dtype=np.float32)
    
    for i in range(n_faces):
        start_idx = face_offsets[i]
        num_nodes = faces[start_idx]
        
        if num_nodes < 3:
            continue # A face must have at least 3 nodes
            
        # 1. Calculate Centroid
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
        
        # 2. Triangulate and accumulate area/normals
        total_area = 0.0
        nx_sum, ny_sum, nz_sum = 0.0, 0.0, 0.0
        
        for j in range(num_nodes):
            idx1 = faces[start_idx + 1 + j]
            idx2 = faces[start_idx + 1 + ((j + 1) % num_nodes)]
            
            p1x = nodes[idx1, 0] - cx
            p1y = nodes[idx1, 1] - cy
            p1z = nodes[idx1, 2] - cz
            
            p2x = nodes[idx2, 0] - cx
            p2y = nodes[idx2, 1] - cy
            p2z = nodes[idx2, 2] - cz
            
            # Cross product
            cx_val = p1y * p2z - p1z * p2y
            cy_val = p1z * p2x - p1x * p2z
            cz_val = p1x * p2y - p1y * p2x
            
            # Magnitude of cross product is 2x Triangle Area
            mag_sq = cx_val*cx_val + cy_val*cy_val + cz_val*cz_val
            if mag_sq > 1e-12:
                mag = np.sqrt(mag_sq)
                tri_area = 0.5 * mag
                total_area += tri_area
                
                # Normalize cross product to get sub-normal, weighted by area down the line
                nx_sum += cx_val / mag * tri_area
                ny_sum += cy_val / mag * tri_area
                nz_sum += cz_val / mag * tri_area
                
        areas[i] = total_area
        
        # Normalize the final accumulated area-weighted normal
        len_sq = nx_sum*nx_sum + ny_sum*ny_sum + nz_sum*nz_sum
        if len_sq > 1e-12:
            length = np.sqrt(len_sq)
            normals[i, 0] = nx_sum / length
            normals[i, 1] = ny_sum / length
            normals[i, 2] = nz_sum / length
            
    return areas, normals, centroids


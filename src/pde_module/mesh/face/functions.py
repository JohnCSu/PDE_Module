import numpy as np
import numba as nb
from pde_module.mesh.utils import getIDs,check_IDs,flatten_and_filter_2D_array,sort_rows
from pde_module.mesh.cell import Cells
from pde_module.mesh.cell_types.cell_types import LOCALFACEORDERING_DICT,CELLTYPES_DICT,LOCAL_EDGE_ORDERING_DICT

def get_faces(cells:Cells,):
    '''
    We need to get calculate all the "faces" for 2D and 3D
    '''
    max_num_faces = cells.int_dtype(max([CELLTYPES_DICT[key].num_faces for key in cells.unique_cell_types]))
    max_num_nodes = cells.int_dtype(max([CELLTYPES_DICT[key].max_nodes_per_face for key in cells.unique_cell_types]))
    
    cell_face_array = _get_3d_facets_(cells.connectivity,cells.IDs,cells.types,max_num_faces,max_num_nodes,LOCALFACEORDERING_DICT)
    
    (unique_faces,face_ids) = get_unique_faces(cell_face_array)
    return unique_faces,face_ids





@nb.njit(cache = True)
def _get_2d_facets_(cells_connectivity,cellIDs,cellTypes,max_num_faces,max_num_nodes,edge_ordering:dict):
    cell_face_array = np.full((len(cellIDs),max_num_faces,max_num_nodes),-1,dtype=cells_connectivity.dtype)
    for i,ID in enumerate(cellIDs):
        num_nodes = cells_connectivity[ID]
        
        nodes = cells_connectivity[ID+1:ID+1+num_nodes]
        celltype_id = cellTypes[i]
        local_face_ordering = edge_ordering[celltype_id]
        num_faces = len(local_face_ordering)
        k = 1
        for j in range(num_faces):
            num_nodes= local_face_ordering[k]
            face_order =  local_face_ordering[k+1:k+1+num_nodes]
            face = nodes[face_order]        
            cell_face_array[i,j,:len(face_order)] = face
            k += 1 + num_nodes
            
    return cell_face_array



@nb.njit(cache = True)
def _get_3d_facets_(cells_connectivity,cellIDs,cellTypes,max_num_faces,max_num_nodes,face_ordering:dict):
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
    unique_faces = np.ascontiguousarray(faces_array[unique_indices],dtype= cell_face_array.dtype) 
    # cell_to_face = unique_inverse.reshape(cell_face_array.shape[0:2])
    if np.sum(unique_faces[0]) == -unique_faces.shape[-1]:# Remove the -1,-1,-1,-1... row if we have multiple cell 
        unique_faces = unique_faces[1:]
        # cell_to_face -= 1 #
         
    # print(cell_to_face)
    # cell_to_face,cell_to_face_ids = flatten_and_filter_2D_array(cell_to_face)
    unique_faces,face_ids = flatten_and_filter_2D_array(unique_faces)
    return (unique_faces,face_ids)
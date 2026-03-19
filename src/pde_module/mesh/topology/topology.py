import numpy as np
import numba as nb
from dataclasses import dataclass

from numba import types

@dataclass
class Flat_array_container:
    array: np.ndarray
    offsets: np.ndarray | None

class Topology:
    '''Describes the connection between shapes of different dimensions e..g face to cell'''
    face_to_cell: Flat_array_container
    cell_to_face: Flat_array_container
    cell_to_cell: Flat_array_container
    def add(self,name,array,ids= None):
        setattr(self,name,Flat_array_container(array,ids))
    
    
@nb.njit(cache = True)
def get_face_to_cell(cell_to_face,cell_to_face_offsets,num_unique_faces):
    face_to_cell = np.full((num_unique_faces,2),-1,dtype=cell_to_face.dtype)
    
    for cell_id in range(len(cell_to_face_offsets)):
        offset = cell_to_face_offsets[cell_id]
        num_faces = cell_to_face[offset]
        for j in range(num_faces): # This part can only be done in serial
            faceID = cell_to_face[offset + j + 1]
            idx = np.int32(face_to_cell[faceID,0] != -1) # if True then idx = 1, if False then idx = 0
            face_to_cell[faceID,idx] = cell_id
    return face_to_cell
    

@nb.njit(cache = True)
def get_cell_to_cell(cell_to_face,cell_to_face_offsets,face_to_cell):
    # For now lets just assign -1 to boundary and then reduce
    #This can be optimized to include only
    cell_to_cell =np.full_like(cell_to_face,-1)
    
    for cell_id in range(len(cell_to_face_offsets)):
        offset =cell_to_face_offsets[cell_id]
        num_faces = cell_to_face[offset] 
        cell_to_cell[offset] = num_faces
        
        for j in range(num_faces):
            faceID = cell_to_face[offset + j + 1]    
            idx = np.int32(face_to_cell[faceID,0] == cell_id)
            neighbor_cell_id = face_to_cell[faceID,idx]
            cell_to_cell[offset+j+1] = neighbor_cell_id
    
    return cell_to_cell
                
                    
                 
                
                
        
        

        
        
    
        
     





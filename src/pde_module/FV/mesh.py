from dataclasses import dataclass
from pde_module.mesh import EulerianMesh
from pde_module.mesh.face import Faces
import numpy as np
import warp as wp
from warp.types import vector
from typing import Any

@dataclass
class Interior_Faces:
    face_ids: np.ndarray |wp.array
    ownerNeighbor: np.ndarray|wp.array
    normals: np.ndarray|wp.array
    centroids: np.ndarray|wp.array

    
    def __init__(self,faces:Faces):
        interior_only_mask = faces.ownerNeighbor[:,-1] != -1
        self.face_ids = np.nonzero(interior_only_mask)[0]
        self.ownerNeighbor = faces.ownerNeighbor[interior_only_mask,:]
        self.normals = faces.normals[interior_only_mask,:]
        self.centroids = faces.centroids[interior_only_mask,:]
        
    def to_warp(self,cell_centroids:wp.array[wp.vec3f],float_dtype = wp.float32):
        self.face_ids = wp.array(self.face_ids,dtype=int)
        self.ownerNeighbor = wp.array(self.ownerNeighbor,dtype=wp.vec2i)
        self.normals = wp.array(self.normals,dtype = vector(3,float_dtype))
        self.centroids = wp.array(self.centroids,dtype = vector(3,float_dtype))
    
    def __len__(self):
        return len(self.face_ids)
class Exterior_Faces:
    face_ids: np.ndarray
    cell_ids: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray  
    groups:dict[str,np.ndarray] 
    def __init__(self,faces:Faces):
        exterior_only_mask = faces.ownerNeighbor[:,-1] == -1
        self.face_ids = np.nonzero(exterior_only_mask)[0]
        self.cell_ids = faces.ownerNeighbor[exterior_only_mask,0]
        self.normals = faces.normals[exterior_only_mask,:]
        self.centroids = faces.centroids[exterior_only_mask,:]
        # self.inv_dist = np.zeros_like(self.centroids)
        self.groups = {'ALL':np.arange(len(self),dtype=np.int32)}
        
        
    
    
    def to_warp(self,cell_centroids:wp.array[wp.vec3f],float_dtype = wp.float32):
        self.face_ids = wp.array(self.face_ids,dtype=int)
        self.cell_ids = wp.array(self.cell_ids,dtype=int)
        self.normals = wp.array(self.normals,dtype = vector(3,float_dtype))
        self.centroids = wp.array(self.centroids,dtype = vector(3,float_dtype))

    def __len__(self):
        return len(self.face_ids)




@wp.kernel
def compute_external_inv_dist(face_centroids:wp.array1d[wp.vec3f],
                              cell_centroids:wp.array1d[wp.vec3f],
                              face_to_cell_ids:wp.array1d[int],
                              inv_dist:wp.array1d[float]):
    tid = wp.tid() # loop faces
    face_centroid = face_centroids[tid]
    cell_id = face_to_cell_ids[tid]
    cell_centroid = cell_centroids[cell_id]
    inv_dist[tid] =  float(1.)/wp.length(face_centroid - cell_centroid)
    
@wp.kernel
def compute_internal_inv_dist(cell_centroids:wp.array1d[wp.vec3f],
                              face_to_cell_ids:wp.array1d[wp.vec2i],
                              inv_dist:wp.array1d[float]):
    tid = wp.tid() # loop faces
    ownerNeighbor = face_to_cell_ids[tid]
    ownerID = ownerNeighbor[0]
    neighborID = ownerNeighbor[1]
    inv_dist[tid] =  float(1.)/wp.length(cell_centroids[neighborID] - cell_centroids[ownerID])
    


class FiniteVolumeMesh(EulerianMesh):
    interior_faces: Interior_Faces
    exterior_faces: Exterior_Faces
    cell_centroids: wp.array
    cell_volumes: wp.array
    face_centroids:wp.array
    face_normals:wp.array
    is_warped: bool = False
    def __init__(self, nodes, cells_connectivity, cell_types, dimension=None, float_dtype=np.float32, int_dtype=np.int32):
        super().__init__(nodes, cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
        self.interior_faces = Interior_Faces(self.faces)
        self.exterior_faces = Exterior_Faces(self.faces)
        
    def to_warp(self):
        if not self.is_warped:
            self.is_warped = True
            
            self.face_normals = wp.array(self.faces.normals,dtype= wp.vec3f) 
            self.cell_centroids = wp.array(self.cells.centroids,dtype= wp.vec3f)
            self.cell_volumes = wp.array(self.cells.volumes)
            self.neighbors = wp.array(self.cells.neighbors,dtype = wp.vec2i)
            self.neighbors_offset = wp.array(self.cells.neighbors_offset)
            self.interior_faces.to_warp(self.cell_centroids)
            self.exterior_faces.to_warp(self.cell_centroids)
            
    
    
    def create_cell_field(self,num_vars,IC = None,output_array = 'warp',device = None,**kwargs):
        shape = (num_vars,len(self.cells))
        arr = np.zeros(shape,dtype =self.float_dtype)
        
        if IC is not None:
            arr:np.ndarray = IC(self.cells.centroids,**kwargs)
            assert arr.shape == shape and isinstance(arr,np.ndarray)
        
        
        match output_array:
            case 'numpy':
                return arr
            case 'warp':
                return wp.array(arr,device= device)
            case _:
                raise ValueError
        
        

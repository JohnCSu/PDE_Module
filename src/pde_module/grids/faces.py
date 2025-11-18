
import warp as wp
import numpy as np
from math import prod
import numba as nb
class Faces():
    dimension:int
    grid_cell_shape:tuple[int]
    normals: wp.array
    
    owners:wp.array
    
    num_internal_faces:int = 0 
    
    def __init__(self,grid_origin,dx,grid_cell_shape,dimension):
        self.dimension = dimension
        self.grid_cell_shape = grid_cell_shape
        self.grid_origin = grid_origin
        self.dx = dx

        self.face_centroid_limits = tuple(  np.linspace(o + self.dx/2.,o+self.dx*(s-1) - self.dx/2,s) if s > 1 else np.array([o]) for o,s in zip(self.grid_origin,self.grid_cell_shape))
        self.num_cells = prod(self.grid_cell_shape)
        self.cellIDs = np.arange(self.num_cells).reshape(self.grid_cell_shape)
        
        self.boundary_groups = dict()    
        
        
        self.ownerNeighbors = -1*np.ones(shape =(self.num_faces,2),dtype = np.int32)
        self.wallIDs = np.empty(shape=self.num_faces,dtype=np.int32)
        self.normals = self.set_normals()
        self.calculate_boundary_Faces()
    
    def set_normals(self):
        normals = np.zeros(shape = (self.dimension,2,self.dimension),dtype= np.float32)
        for axis in range(self.dimension):
            for j,normal in enumerate([-1.,1.]):
                    normals[axis,j,axis] = normal
        
        return normals.reshape((-1,self.dimension))    
        
    @property
    def area(self):
        return self.dx**self.dimension
    
    @property
    def num_faces(self):
        num_faces = 0
        
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] +=1
            num_faces += prod(faces_shape)
        return num_faces
    
    @property
    def num_boundary_faces(self):
        
        num_faces = 0
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] = 1 
            num_faces += prod(faces_shape)*2

        return num_faces
    
    @property
    def num_internal_faces(self):
        return self.num_faces - self.num_boundary_faces
        
        
    def calculate_boundary_Faces(self):
        
        boundary_face_owners = self.ownerNeighbors
            
        initial = 0
        
        
        for axis,axis_Name in zip(range(self.dimension),['X','Y','Z']):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] = 1 
            num_faces = prod(faces_shape)
            
            for j,(fixed_point,side) in enumerate(zip([0,self.grid_cell_shape[axis]-1],['-','+'])):
                indices = [slice(None),slice(None),slice(None)]
                indices[axis] = fixed_point
                boundary_face_owners[initial:initial+num_faces,0] = self.cellIDs[tuple(indices)].squeeze().flatten() # tuple indexing (each element corresponds to element) is different to list indexing with numpy arrays
                self.wallIDs[initial:initial+num_faces] = axis*2 + j
                group_name = side +axis_Name 
                self.boundary_groups[group_name] = np.arange(0,num_faces) + initial
                
                initial += num_faces
        self.boundary_ownerNeighbors = self.ownerNeighbors[:initial]
        self.boundary_groups['ALL'] = np.arange(self.num_boundary_faces,dtype=np.int32)
        
    def calculate_internal_Faces(self):
        
        # internal_face_owners = -1*np.ones(shape = self.num_internal_faces,dtype= np.int32)
        internal_face_owners = self.ownerNeighbors
        
        intital = self.num_boundary_faces
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] -= 1 
            num_faces = prod(faces_shape)
            
            indices = [slice(None),slice(None),slice(None)]
            indices[axis] = slice(0,-1)
            
            internal_face_owners[intital:intital+num_faces,0] = self.cellIDs[tuple(indices)].squeeze().flatten()
            indices[axis] = slice(1,None)
            internal_face_owners[intital:intital+num_faces,1] = self.cellIDs[tuple(indices)].squeeze().flatten()
            
            intital += num_faces
            
        self.internal_ownerNeighbors = self.ownerNeighbors[self.num_boundary_faces:intital]
            
        
    def create_faces_field(self):
        pass 
    
if __name__ == '__main__':
    f = Faces((0.,0.,0.),dx = 0.5,grid_cell_shape=(3,3,2),dimension=3)
    
    
    f.calculate_boundary_Faces()
    f.calculate_internal_Faces()
    print(f.ownerNeighbors)
    print(f.num_faces,f.num_boundary_faces,f.num_internal_faces)
    print(f.normals)
    
    
    
    
    
    
    
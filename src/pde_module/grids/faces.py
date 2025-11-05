
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
        
        
        
        self.ownnerNeighbors = -1*np.ones(shape =(self.num_faces,2),dtype = np.int32)
        self.normals_mapping = np.empty(shape=self.num_faces,dtype=np.float32)
        self.normals = np.zeros(shape = (2*self.dimension,self.dimension),dtype= np.float32)
        
        
        
        
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
        
        boundary_face_owners = self.ownnerNeighbors
            
        initial = 0
        for axis,axis_Name in zip(range(self.dimension),['X','Y','Z']):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] = 1 
            num_faces = prod(faces_shape)
            
            for fixed_point,side in zip([0,self.grid_cell_shape[axis]-1],['-','+']):
                indices = [slice(None),slice(None),slice(None)]
                indices[axis] = fixed_point
                boundary_face_owners[initial:initial+num_faces,0] = self.cellIDs[tuple(indices)].squeeze() # tuple indexing (each element corresponds to element) is different to list indexing with numpy arrays
                
                group_name = side +axis_Name 
                self.boundary_groups[group_name] = np.arange(0,num_faces) + initial
                
                initial += num_faces
        
        
    def calculate_internal_Faces(self):
        
        # internal_face_owners = -1*np.ones(shape = self.num_internal_faces,dtype= np.int32)
        internal_face_owners = self.ownnerNeighbors
        
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
            

            
        
    def create_faces_field(self):
        pass 
    
if __name__ == '__main__':
    f = Faces((0.,0.,0.),dx = 0.5,grid_cell_shape=(3,3,1),dimension=2)
    
    
    f.calculate_boundary_Faces()
    f.calculate_internal_Faces()
    print(f.ownnerNeighbors)
    print(f.num_faces,f.num_boundary_faces,f.num_internal_faces)
    # f.calculate_boundary_Faces()
    # # f.calculate_internal_Faces()
    # print(f.boundary_cells)
    # print(f.owners)
    # print()
    
    # Nx, Ny, Nz = (3,3,1)


    # g = (3,3,1)
    # # total cells
    # Nc = Nx * Ny * Nz
    # ijk = np.indices((Nx,Ny,Nz),dtype = np.int32)
    # # ijk = np.moveaxis(ijk,0,-1) # Have the indices axis at end instead
    # cells_ijk = np.moveaxis(ijk,0,-1).reshape(-1,3)
    
    
    # x_owner = cells_ijk[:,]
    # # x faces
    # x_owner = ijk[:,:-1,:,:].reshape(3,-1)
    # x_neighbor = ijk[:,1:,:,:].reshape(3,-1)
    # print(x_owner)
    
    # x_owner = np.ravel_multi_index(x_owner,dims = g)
    # x_neighbor = np.ravel_multi_index(x_neighbor,dims = g)
    # print(x_owner)
    # print(x_neighbor)
    # print('hi')
    # print(cells_ijk)
    
    
    
    
    
    
    
    

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


    @property
    def area(self):
        return self.dx**self.dimension
    
    @property
    def num_faces(self):
        return self.num_boundary_faces + self.num_internal_faces
    
    @property
    def num_boundary_faces(self):
        
        num_faces = 0
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] = 1 
            num_faces += prod(faces_shape)*2

        return num_faces
    
    
    def calculate_boundary_Faces(self):        
        boundary_face_owners = -1*np.ones(shape = (self.num_boundary_faces,3),dtype=np.int32)
        
        
        initial = 0
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] = 1 
            num_faces = prod(faces_shape)
            for fixed_point in [0,self.grid_cell_shape[axis]-1]:
                indices = np.indices(faces_shape,dtype= np.int32)
                indices = np.moveaxis(indices,0,-1).reshape(-1,3)
                indices[:,axis] += fixed_point
                
                boundary_face_owners[initial:initial+num_faces] = indices
                initial += num_faces
        
        self.boundary_cells,self.owners = np.unique(boundary_face_owners,return_inverse=True,axis = 0)
        
    def calculate_internal_Faces(self):
        
        
        self.cell_indices = np.indices(self.grid_cell_shape,dtype=int).reshape(-1,3)
        
        for axis in range(self.dimension):
            faces_shape = list(self.grid_cell_shape)
            faces_shape[axis] -= 1 
            num_faces = prod(faces_shape)
            
            faces = np.zeros(shape = (num_faces,2),dtype= np.int32)
            
            
        
    def create_faces_field(self):
        pass 
    
if __name__ == '__main__':
    f = Faces((0.,0.,0.),dx = 0.5,grid_cell_shape=(3,3,1),dimension=2)
    
    f.calculate_boundary_Faces()
    # f.calculate_internal_Faces()
    print(f.boundary_cells)
    print(f.owners)
    print()
    
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
    
    
    
    
    
    
    
    
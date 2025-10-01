import warp as wp
import numpy as np
from math import prod



class Cells():
    volume:np.ndarray | wp.array
    shape:tuple[int]
    dimension:int
    

class Strucutred3D(Cells):
    '''
    Stores info for structured non-uniform cells 
    '''
    def __init__(self,x,y,z,dimension):
        assert dimension == 3
        self.dimension = dimension
        
        self.cell_center_x = self.get_midpoints(x)
        self.cell_center_y = self.get_midpoints(y)
        self.cell_center_z = self.get_midpoints(z)
        self.volume = self.get_cell_volume()
        self.cell_center_coordinate_vectors = [self.cell_center_x,self.cell_center_y,self.cell_center_z]
        self.shape = tuple(len(coord) for coord in self.cell_center_coordinate_vectors)
        
    @staticmethod
    def get_midpoints(arr):
        if len(arr) == 1:
            return arr
        return (arr[:-1] + arr[1:])/2.

    
    def get_cell_volume(self):
        
        diffs = []
        for i,coord_array in enumerate(self.cell_center_coordinate_vectors[:self.dimension]):
            
            shape = [1 for _ in range(self.dimension)]
            shape[i] = -1
            
            diffs.append(np.diff(coord_array).reshape(shape))
        
        return prod(diffs)





class StructuredFaces(Cells):
    def __init__(self,x,y,z,dimension):
        assert dimension == 3 # For now
        self.face_center_x = x
        self.face_center_y = y
        self.face_center_z = z
    
    
    def get_cell_
    
    
class CellGrid:
    '''
    Define the nodes along each axis. Must be 3D
    '''
    def __init__(self,x,y=None,z=None,float_dtype = np.float32):
        
        assert isinstance(x,np.ndarray)
        self.wp_float_dtype = wp.dtype_from_numpy(float_dtype)
        
        arrs = []
        for axis in [x,y,z]:
            if axis is None:
                arr = np.array([0],dtype=float_dtype)
            else:
                arr = axis
                assert len(arr.shape) == 1 and len(arr) >=3, 'There must be at least 3 points along a specified axis'
            
            arrs.append(arr) 
                
        
        self.dimension = sum([1 for i in arrs if len(i) > 1])
        
        self.float_dtype = float_dtype
        
        
        self.coordinate_vectors = arrs     
        '''tuple of coordinate vectors along each axis that define the grid excluding ghost cells'''
        
        self.x,self.y,self.z = self.coordinate_vectors
        self.shape = tuple(len(arr) for arr in self.coordinate_vectors)
        '''Shape of grid excluding any ghost points'''
        
        '''We only use neighbors'''
        self.levels = 1
        
        
import numpy as np
import warp as wp
from dataclasses import dataclass, field
from warp.types import vector,matrix
from pde_module.utils.types import wp_Array,wp_Vector
from pde_module.LBM.utils import array_to_vec_or_mat,vec_or_mat_to_numpy

def find_opposite_indices(int_directions):
    # Instead we should convert the rows into a single byte dtype and search for that
    opposite_indices = np.zeros(len(int_directions),dtype= int_directions.dtype)
    # Turn into a list of tuples
    directions = [tuple(row) for row in int_directions]
    for i,direction in enumerate(int_directions): # n**2 but directions is small so ok
        opposite_indices[i] = directions.index(tuple(-direction))
    return opposite_indices

@dataclass
class LatticeModel:
    directions:np.ndarray | wp_Array
    weights:np.ndarray | wp_Array
    dimension:int
    int_dtype:np.dtype  = np.int32
    float_dtype:np.dtype = np.float32
    
    int_directions:np.ndarray | wp_Array = field(init= False)
    '''(N,D) array of Directions expressed as integer dtype. Used for indexing ops'''
    float_directions:np.ndarray | wp_Array = field(init= False)
    '''(N,D) array of Directions expressed as floating point dtype. Used in collision step'''
    sides: np.ndarray | wp_Array = field(init=False)
    '''(D,2,L) array of the directions associetd with each face. First axis is ordered as X,Y,Z then 2nd is -side and then +side'''
    opposite_indices:np.ndarray = field(init=False)
    num_distributions:int = field(init=False)
    num_directions_per_side: int = field(init=False)
    backend:str = field(init=False,default='numpy')
    
    
    def __post_init__(self):
        # Assumes Isotropic i.e each face should have the same number of discrete velocites in its normal direction
        self.weights = np.astype(self.weights,self.float_dtype)
        self.int_directions = np.astype(self.directions,self.int_dtype)
        self.float_directions = np.astype(self.directions,self.float_dtype)
        
        assert self.int_directions.shape[-1] == self.dimension
        
        self.num_distributions = len(self.int_directions)
        self.num_directions_per_side = len(np.nonzero(self.int_directions[:,0] == 1)[0])
        self.sides = np.zeros((self.dimension,2,self.num_directions_per_side),self.int_dtype)
        self.opposite_indices = find_opposite_indices(self.int_directions)
        
        
        for axis in range(self.dimension): # Axis
            for j,side in enumerate([-1,1]): # is pos or neg
                self.sides[axis,j] = np.nonzero(self.int_directions[:,axis] == side)[0]
                
    def to_warp(self):
        if self.backend != 'warp':
            self.int_dtype,self.float_dtype = wp.dtype_from_numpy(self.int_dtype),wp.dtype_from_numpy(self.float_dtype)
            for attr,value in self.__dict__.items():
                match attr:
                    case 'sides':
                        self.sides = wp.array(self.sides,dtype=vector(self.num_directions_per_side,self.int_dtype))
                    case 'int_directions' | 'float_directions' |'weights'| "opposite_indices": # Convert to vector | matrix type
                        setattr(self,attr,array_to_vec_or_mat(value))
                    case _:
                        if isinstance(value,np.ndarray):
                            setattr(self,attr,wp.array(value,dtype=wp.dtype_from_numpy(value.dtype)))
                            
            self.backend = 'warp'    
    
    def to_numpy(self):
        if self.backend != 'numpy':
            self.int_dtype,self.float_dtype = wp.dtype_to_numpy(self.int_dtype),wp.dtype_to_numpy(self.float_dtype)
            for attr,value in self.__dict__.items():
                match attr:
                    case 'int_directions' | 'float_directions' | "opposite_indices" |'weights': 
                        setattr(self,attr,vec_or_mat_to_numpy(value))
                    case _:
                        if wp.types.is_array(value):
                            setattr(self,attr,value.numpy())
            self.backend = 'numpy'






if __name__ == '__main__':    
    d2q9_directions = np.array([
        [ 0,  0], # 0: Center (rest)
        [ 1,  0], # 1: East
        [ 0,  1], # 2: North
        [-1,  0], # 3: West
        [ 0, -1], # 4: South
        [ 1,  1], # 5: North-East
        [-1,  1], # 6: North-West
        [-1, -1], # 7: South-West
        [ 1, -1]  # 8: South-East
    ])
    
    d2q9_weights = np.array([
    4/9,                          # 0: Center
    1/9, 1/9, 1/9, 1/9,           # 1-4: Axis
    1/36, 1/36, 1/36, 1/36        # 5-8: Diagonal
])
    
    D2Q9 = LatticeModel(d2q9_directions,d2q9_weights,2)
    
    
    print(D2Q9.opposite_indices)        
    D2Q9.to_warp()
    # print(D2Q9)
    D2Q9.to_numpy()
    # print(D2Q9)
    assert True # Completed
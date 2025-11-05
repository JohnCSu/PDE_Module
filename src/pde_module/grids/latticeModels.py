import warp as wp
import numpy as np
from typing import Any

class LatticeModel:
    dimension:int
    float_dtype: Any
    num_discrete_velocities:int
    velocity_directions_int:wp.Matrix
    velocity_directions_float:wp.Matrix
    weights:wp.Vector
    opposite_indices:wp.Vector
    wall_indices:wp.Matrix
    

class D2Q9_Model:
    def __init__(self):
        self.dimension = 2
        self.float_dtype = np.float32
        
        self.num_discrete_velocities = 9
        
        self.D2Q9_velocities = np.array([
            [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
            [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
        ]).swapaxes(0,1)
        self.velocity_directions_int =wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = int)(self.D2Q9_velocities)
        
        self.velocity_directions_float = wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = float)(self.D2Q9_velocities.astype( self.float_dtype))
        
        
        self.indices = wp.vec(length = self.num_discrete_velocities,dtype = int)(np.arange(self.num_discrete_velocities,dtype= int))
        
        self.weights = wp.vec(length = self.num_discrete_velocities,dtype = wp.dtype_from_numpy(self.float_dtype) )([
                                4/9,                        # Center Velocity [0,]
                                1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
                                1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
                                ])

        self.opposite_indices = wp.vec(length = self.num_discrete_velocities,dtype = np.int32)([
            0, 3, 4, 1, 2, 7, 8, 5, 6,
            ])
        
        self.wall_indices = wp.mat(shape = (4,3),dtype = int)([
        [3, 6, 7], #Left
        [1, 5, 8], #Right
        [2, 5, 6], #Up
        [4, 7, 8], #down
        ])
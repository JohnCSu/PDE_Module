from .latticeModel import LatticeModel
import numpy as np

def D2Q9(int_dtype=np.int32,float_dtype=np.float32):
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
    ],dtype=int_dtype)
    
    d2q9_weights = np.array([
    4/9,                          # 0: Center
    1/9, 1/9, 1/9, 1/9,           # 1-4: Axis
    1/36, 1/36, 1/36, 1/36        # 5-8: Diagonal
    ],float_dtype)
    
    return LatticeModel(d2q9_directions,d2q9_weights,2,int_dtype,float_dtype)
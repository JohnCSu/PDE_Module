import numpy as np


class Edges:
    '''
        Container to store Edge information
        Edges is stored as a (E,2) numpy array of node IDs
        '''
    connectivity:np.ndarray
    lengths: np.ndarray
    float_dtype: np.ndarray
    int_dtype: np.ndarray
    def __init__(self,connectivity,float_dtype = np.float32,int_dtype = np.int32):
        
        self.connectivity = connectivity
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype


    
    
            
        
        
        
        
        
        
        
    
    



        
        
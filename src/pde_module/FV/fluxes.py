from pde_module.mesh import Mesh
from pde_module.mesh.topology import *
import numpy as np



'''
For FV we need to know faces and the
'''



class FiniteVolumeMesh(Mesh):
    def __init__(self, nodes, cells_connectivity, cell_types, dimension = None, float_dtype=np.float32, int_dtype=np.int32):
        super().__init__(nodes, cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
        
        
        
        
        
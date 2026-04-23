from pde_module.stencil import Stencil
import numpy as np
import warp as wp
from pde_module.FV.mesh import FiniteVolumeMesh

class FiniteVolume(Stencil):
    '''
    Base Module Class for FiniteVolume
    '''
    def __init__(self,mesh:FiniteVolumeMesh ,float_dtype=wp.float32):
        super().__init__()
        self.float_dtype = float_dtype
        self.mesh = mesh

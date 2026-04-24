from pde_module.stencil import Stencil
import numpy as np
import warp as wp
from pde_module.FV.mesh import FiniteVolumeMesh

class FiniteVolume(Stencil):
    '''
    Base Module Class for FiniteVolume
    '''
    def __init__(self,mesh:FiniteVolumeMesh ,float_dtype=None):
        super().__init__()
        if float_dtype is None:
            self.float_dtype = wp.dtype_from_numpy(self.mesh.float_dtype)
        else:     
            self.float_dtype = float_dtype
            if self.float_dtype != self.mesh.float_dtype:
                Warning(f'Specified Float Dtype for module was {float_dtype} which is different to the mesh dtype of {wp.dtype_from_numpy(mesh.float_dtype)}')

        self.mesh = mesh

from pde_module.stencil import Stencil
import numpy as np
import warp as wp

class FiniteVolume(Stencil):
    def __init__(self,float_dtype = wp.float32):
        super().__init__()
        self.float_dtype = float_dtype

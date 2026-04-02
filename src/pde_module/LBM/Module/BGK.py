from .LBM_Stencil import LBM_Stencil
import numpy as np
import warp as wp
from warp.types import vector
from pde_module.utils.types import wp_Array, wp_Vector, wp_Matrix
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.Kernel import create_BGK_collision
from pde_module.LBM.Functional import bgk_collision


class BGK_collision(LBM_Stencil):
    def __init__(self, latticeModel: LatticeModel, grid_shape: tuple[int]):
        super().__init__(latticeModel, grid_shape)

    def __call__(self, f_in, tau):
        return super().__call__(f_in, tau)

    @setup
    def initialise(self, f_in, tau):
        self.latticeModel.to_warp()
        self.kernel = create_BGK_collision(
            self.latticeModel.float_directions,
            self.latticeModel.weights,
            self.num_distributions,
            self.dimension,
            self.grid_shape,
            self.latticeModel.float_dtype,
        )
        self.f_out = self.create_output_array(f_in)

    def forward(self, f_in, tau):
        return bgk_collision(self.kernel, f_in, tau, self.f_out, device=self.device)

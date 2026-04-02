from .LBM_Stencil import LBM_Stencil
import warp as wp
from pde_module.utils.types import wp_Array
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.Kernel import create_streaming_kernel
from pde_module.LBM.Functional import streaming


class Streaming(LBM_Stencil):
    grid_shape: tuple[int]

    def __init__(self, latticeModel: LatticeModel, grid_shape: tuple[int]):
        super().__init__(latticeModel, grid_shape)

    def __call__(self, f_in: wp_Array):
        return super().__call__(f_in)

    @setup
    def initialise(self, f_in: wp_Array):
        assert f_in.shape[0] == self.num_distributions
        self.latticeModel.to_warp()
        self.kernel = create_streaming_kernel(
            f_in,
            self.latticeModel.int_directions,
            self.num_distributions,
            self.grid_shape,
            self.dimension,
        )
        self.f_out = self.create_output_array(f_in)

    def forward(self, f_in: wp_Array) -> wp_Array:
        return streaming(
            self.kernel, f_in, self.f_out, self.grid_shape, device=self.device
        )

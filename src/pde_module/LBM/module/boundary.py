from .LBM_Stencil import LBM_Stencil
import numpy as np
import warp as wp
from warp.types import vector
from pde_module.utils.types import wp_Array, wp_Vector, wp_Matrix, Any
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.stencil.hooks import *
from pde_module.LBM.kernel import create_boundary_kernel
from pde_module.LBM.functional import boundary

FLUID = 0
SOLID_WALL = 1
MOVING_WALL = 2
EQUILIBRIUM = 3


class Boundary(LBM_Stencil):
    sigma: float = 0.1
    flags: np.ndarray
    groups: dict[str, Any]

    @classmethod
    def from_LBM_Mesh(cls, mesh):
        return super().from_LBM_Mesh(mesh, "flags", "groups")

    def __init__(self, latticeModel, grid_shape, flags, groups):
        super().__init__(latticeModel, grid_shape)
        self.flags = flags
        self.BC_velocity = np.full(
            self.grid_shape + (self.dimension,),
            np.nan,
            dtype=self.latticeModel.float_dtype,
        )
        self.BC_density = np.full(
            self.grid_shape, np.nan, dtype=self.latticeModel.float_dtype
        )
        self.groups = groups

    def set_BC(
        self,
        ids: str | tuple[np.ndarray | int | slice],
        boundary_type,
        velocity=None,
        density=None,
    ):
        if boundary_type == 2 or boundary_type == 3:
            assert velocity is not None or density is not None

        match ids:
            case str():
                assert ids in self.groups.keys()
                ids = self.groups[ids]
            case tuple():
                assert len(ids) == 3
                assert all(isinstance(obj, (np.ndarray, int, slice)) for obj in ids)
            case _:
                raise TypeError("Strings or tuples of ndarrays are allowed")

        self.flags[*ids] = boundary_type
        if velocity is not None:
            self.BC_velocity[*ids, :] = velocity
        if density is not None:
            self.BC_density[ids] = density

    @setup
    def initialize(self, f_in):
        self.latticeModel.to_warp()
        self.indices = np.nonzero(self.flags.ravel())[0].astype(
            wp.dtype_to_numpy(self.latticeModel.int_dtype)
        )
        self.warp_indices = wp.array(self.indices)
        self.warp_flags = wp.array(self.flags, dtype=wp.uint8)

        self.warp_BC_velocity = wp.array(
            self.BC_velocity.reshape((-1, self.dimension))[self.indices, :],
            dtype=vector(self.dimension, self.latticeModel.float_dtype),
        )
        self.warp_BC_density = wp.array(self.BC_density.ravel()[self.indices])

        self.kernel = create_boundary_kernel(
            self.num_distributions,
            self.latticeModel.opposite_indices,
            self.latticeModel.int_directions,
            self.latticeModel.float_directions,
            self.latticeModel.weights,
            self.dimension,
            self.grid_shape,
            self.sigma,
            self.latticeModel.float_dtype,
        )

        self.f_out = self.create_output_array(f_in)

    def forward(self, f_in):
        return boundary(
            self.kernel,
            f_in,
            self.warp_flags,
            self.warp_indices,
            self.warp_BC_velocity,
            self.warp_BC_density,
            self.f_out,
            device=self.device,
        )

from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp

from pde_module.stencil.hooks import *
from pde_module.FV.kernel import create_advection_kernel
from pde_module.FV.functional import advection


class Advection(FiniteVolume):
    def __init__(self, mesh, interpolation="upwind", float_dtype=wp.float32):
        super().__init__(mesh, float_dtype)
        self.interpolation = interpolation

    def __call__(
        self,
        scalar_cell_field,
        scalar_boundary_values,
        velocity_cell_field,
        velocity_boundary_values,
        density=1.0,
    ):
        return super().__call__(
            scalar_cell_field,
            scalar_boundary_values,
            velocity_cell_field,
            velocity_boundary_values,
            density,
        )

    @setup
    def initialise(
        self,
        scalar_cell_field,
        scalar_boundary_values,
        velocity_cell_field,
        velocity_boundary_values,
        density,
    ):
        self.mesh.to_warp()
        self.field_shape = scalar_cell_field.shape
        self.num_vars = self.field_shape[0]
        self.num_cells = self.field_shape[-1]
        self.internal_kernel, self.external_kernel = create_advection_kernel(
            self.num_vars, self.float_dtype
        )

        self.output_field = wp.empty_like(scalar_cell_field)

    def forward(
        self,
        scalar_cell_field,
        scalar_boundary_values,
        velocity_cell_field,
        velocity_boundary_values,
        density,
    ):
        return advection(
            self.internal_kernel,
            self.external_kernel,
            scalar_cell_field,
            scalar_boundary_values,
            velocity_cell_field,
            velocity_boundary_values,
            density,
            self.mesh.neighbors,
            self.mesh.neighbors_offset,
            self.mesh.face_normals,
            self.mesh.cell_volumes,
            self.mesh.exterior_faces.cell_ids,
            self.mesh.exterior_faces.normals,
            self.output_field,
            self.num_cells,
            len(self.mesh.exterior_faces),
            device=self.device,
        )

from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp

from pde_module.stencil.hooks import *
from pde_module.FV.kernel import create_diffusion_kernel
from pde_module.FV.functional import diffusion


class Diffusion(FiniteVolume):
    def __init__(
        self, mesh: FiniteVolumeMesh, interpolation="central", float_dtype=wp.float32
    ):
        super().__init__(mesh, float_dtype)
        self.mesh = mesh
        self.interpolation_key = interpolation

    def __call__(self, input_field, boundary_values, viscosity):
        return super().__call__(input_field, boundary_values, viscosity)

    @setup
    def initialise(self, input_field, boundary_values, viscosity):
        self.mesh.to_warp()
        self.field_shape = input_field.shape
        self.num_vars = self.field_shape[0]
        self.internal_kernel, self.external_kernel = create_diffusion_kernel(
            self.num_vars, self.float_dtype
        )

        self.output_field = wp.empty_like(input_field)

    def forward(self, input_field, boundary_values, viscosity):
        return diffusion(
            self.internal_kernel,
            self.external_kernel,
            input_field,
            boundary_values,
            viscosity,
            self.mesh.neighbors,
            self.mesh.neighbors_offset,
            self.mesh.face_normals,
            self.mesh.cell_centroids,
            self.mesh.cell_volumes,
            self.mesh.exterior_faces.cell_ids,
            self.mesh.exterior_faces.centroids,
            self.mesh.exterior_faces.normals,
            self.output_field,
            len(self.mesh.neighbors_offset),
            len(self.mesh.exterior_faces),
            device=self.device,
        )

from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp

from pde_module.stencil.hooks import *
from pde_module.FV.kernel import create_Divergence_kernel
from pde_module.FV.functional import divergence


class Divergence(FiniteVolume):
    """
    Only Vectors of size 3 is currently supported
    """

    def __init__(self, mesh, float_dtype=wp.float32):
        super().__init__(mesh, float_dtype)

    def __call__(self, vector_field, boundary_values, alpha=1.0):
        return super().__call__(vector_field, boundary_values, alpha)

    @setup
    def initialise(self, vector_field, boundary_values, alpha):
        self.field_shape = vector_field.shape
        assert self.field_shape[0] == 3, (
            "Only vector fields of size 3 availiable for now"
        )
        assert self.field_shape[0] == boundary_values.shape[0]
        self.num_vars = self.field_shape[0]
        self.num_cells = self.field_shape[-1]
        self.num_boundary_faces = len(self.mesh.exterior_faces)
        self.internal_kernel, self.external_kernel = create_Divergence_kernel(
            self.float_dtype
        )

        self.output_field = wp.empty((1, self.num_cells), dtype=self.float_dtype)

    def forward(self, vector_field, boundary_values, alpha):
        return divergence(
            self.internal_kernel,
            self.external_kernel,
            vector_field,
            boundary_values,
            alpha,
            self.mesh.neighbors,
            self.mesh.neighbors_offset,
            self.mesh.face_normals,
            self.mesh.face_centroids,
            self.mesh.cell_centroids,
            self.mesh.cell_volumes,
            self.mesh.exterior_faces.cell_ids,
            self.mesh.exterior_faces.normals,
            self.output_field,
            self.num_cells,
            self.num_boundary_faces,
            device=self.device,
        )

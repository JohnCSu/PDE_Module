from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp

from pde_module.stencil.hooks import *
from pde_module.FV.kernel import create_grad_kernel
from pde_module.FV.functional import grad


class Grad(FiniteVolume):
    """
    Only Scalar Fields are currently Supported
    """

    def __init__(self, mesh, float_dtype=wp.float32):
        super().__init__(mesh, float_dtype)

    def __call__(self, cell_field, boundary_values, alpha=1.0):
        return super().__call__(cell_field, boundary_values, alpha)

    @setup
    def initialise(self, cell_field, boundary_values, alpha):
        self.mesh.to_warp()

        assert cell_field.shape[0] == 1, "Only scalar fields availiable for now"
        self.field_shape = cell_field.shape
        self.num_vars = cell_field.shape[0]
        self.num_cells = cell_field.shape[-1]
        self.num_boundary_faces = len(self.mesh.exterior_faces)
        self.internal_kernel, self.external_kernel = create_grad_kernel(
            self.float_dtype
        )
        self.output_field = wp.empty((3, self.num_cells), dtype=cell_field.dtype)

    def forward(self, cell_field, boundary_values, alpha):
        return grad(
            self.internal_kernel,
            self.external_kernel,
            cell_field,
            boundary_values,
            alpha,
            self.mesh.neighbors,
            self.mesh.neighbors_offset,
            self.mesh.face_normals,
            self.mesh.cell_volumes,
            self.mesh.exterior_faces.cell_ids,
            self.mesh.exterior_faces.normals,
            self.output_field,
            self.num_cells,
            self.num_boundary_faces,
            device=self.device,
        )

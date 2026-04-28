from pde_module.stencil.hooks import *
from pde_module.utils import get_unique_key
import numpy as np
import warp as wp
from .finiteVolume import FiniteVolume
from ..flags import DIRICHLET, VON_NEUMANN
from ..mesh import FiniteVolumeMesh
from pde_module.FV.kernel import create_FV_Boundary_kernel
from pde_module.FV.functional import boundary


class Boundary(FiniteVolume):
    def __init__(self, mesh: FiniteVolumeMesh, num_vars, float_dtype=wp.float32):
        assert isinstance(mesh, FiniteVolumeMesh)
        super().__init__(mesh, float_dtype)
        self.num_vars = num_vars
        self.num_faces = len(mesh.exterior_faces)
        self.mesh = mesh
        self.boundary_types = np.zeros((num_vars, self.num_faces), dtype=np.uint8)
        self.boundary_values = np.zeros(
            (num_vars, self.num_faces), dtype=wp.dtype_to_numpy(float_dtype)
        )
        self.groups = mesh.exterior_faces.groups

    def set_BC(
        self,
        ids,
        boundary_type,
        boundary_value,
        output_ids=None,
        boundary_group_name=None,
    ):
        match ids:
            case str():
                assert ids in self.groups.keys()
                groupName = ids
                ids = self.groups[ids]
            case _:
                groupName = (
                    get_unique_key(self.groups, base_name="BC_group")
                    if boundary_group_name is None
                    else boundary_group_name
                )
                ids = np.array(ids, dtype=int)
                self.groups[groupName] = ids

        assert isinstance(groupName, str)

        if output_ids is None:
            output_ids = slice(None)

        assert boundary_type in [DIRICHLET, VON_NEUMANN]

        self.boundary_types[output_ids, ids] = boundary_type
        self.boundary_values[output_ids, ids] = boundary_value

    def __call__(self, cell_field):
        return super().__call__(cell_field)

    @setup
    def setup(self, cell_field):
        self.mesh.to_warp()
        self.boundary_types = wp.array(
            self.boundary_types, shape=self.boundary_types.shape
        )
        self.boundary_values = wp.array(
            self.boundary_values, shape=self.boundary_values.shape
        )
        self.kernel = create_FV_Boundary_kernel(self.num_vars, self.float_dtype)
        self.boundary_field = wp.empty_like(self.boundary_values, self.device)

    def forward(self, cell_field):
        exterior = self.mesh.exterior_faces
        return boundary(
            self.kernel,
            cell_field,
            exterior.cell_ids,
            self.boundary_values,
            self.boundary_types,
            exterior.centroids,
            self.mesh.cell_centroids,
            self.boundary_field,
            self.num_faces,
            device=self.device,
        )

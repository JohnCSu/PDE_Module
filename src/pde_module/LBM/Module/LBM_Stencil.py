from pde_module.stencil import Stencil
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.utils.types import wp_Dtype
from pde_module.LBM.mesh import LBM_Mesh


class LBM_Stencil(Stencil):
    latticeModel: LatticeModel
    num_distributions: int
    dimension: int
    grid_shape: tuple[int]

    def __init__(self, latticeModel: LatticeModel, grid_shape, **kwargs):
        super().__init__()
        self.latticeModel: LatticeModel = latticeModel
        self.grid_shape = grid_shape
        self.num_distributions = latticeModel.num_distributions
        self.dimension = latticeModel.dimension

        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_LBM_Mesh(cls, mesh: LBM_Mesh, *args):
        assert isinstance(mesh, LBM_Mesh)
        kwargs = {}
        for key in args:
            assert isinstance(key, str)
            kwargs[key] = getattr(mesh, key)

        return cls(latticeModel=mesh.latticeModel, grid_shape=mesh.grid_shape, **kwargs)

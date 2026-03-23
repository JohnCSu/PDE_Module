from pde_module.stencil import Stencil
from pde_module.LBM.lattticeModels.latticeModel import LatticeModel
from pde_module.utils.dummy_types import wp_Dtype
from pde_module.LBM.mesh import LBM_Mesh
class LBM_Stencil(Stencil):
    latticeModel:LatticeModel
    num_distributions:int
    dimension:int
    grid_shape:tuple[int]
    def __init__(self,latticeModel:LatticeModel,grid_shape):
        super().__init__()
        self.latticeModel:LatticeModel = latticeModel
        self.grid_shape = grid_shape
        self.num_distributions = latticeModel.num_distributions
        self.dimension = latticeModel.dimension
        
    @classmethod
    def from_LBM_Mesh(cls,mesh:LBM_Mesh):
        assert isinstance(mesh, LBM_Mesh)
        return cls(mesh.latticeModel,mesh.grid_shape)
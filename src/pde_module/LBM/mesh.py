from pde_module.mesh import UniformGridMesh
import numpy as np
import warp as wp
from pde_module.LBM.lattticeModels import LatticeModel,D2Q9


def get_latticeModel(latticeModel:str,int_dtype,float_dtype):
    match latticeModel:
        case 'D2Q9':
            return D2Q9(int_dtype,float_dtype)
        case _:
            raise ValueError()


class LBM_Mesh(UniformGridMesh):
    grid_shape: tuple[int] # This is cell NOT node
    '''node Shape of Mesh as 3-tuple'''
    num_cells:int
    latticeModel:LatticeModel
    flags:np.ndarray
    def __init__(self,
                latticeModel:LatticeModel | str,
                dx: float,
                nodes_per_axis: tuple[int, ...],
                origin: np.ndarray | None = None,
                float_dtype=np.float32,
                int_dtype=np.int32,):
        
        
        super().__init__(dx, nodes_per_axis, origin,0, float_dtype, int_dtype)
        
        if isinstance(latticeModel,str):
           latticeModel = get_latticeModel(latticeModel,int_dtype,float_dtype)
        else:
            assert isinstance(latticeModel,LatticeModel)
            
        self.latticeModel = latticeModel
        self.grid_shape = tuple(n if n > 1 else 1 for n in self.nodes_per_axis)
        assert self.latticeModel.dimension == sum(1 for i in nodes_per_axis if i > 1), 'Lattice Model must match dimension of mesh'
        
       
        

    def create_field(self,num_outputs:int,initial_value:float = 0.,backend= 'warp'):
        '''Creates an SoA array of flat array i.e O,C where O is number of outputs and C is number of cells'''
        match backend:
            case 'warp':
                return wp.full((num_outputs,self.num_nodes),value = initial_value,dtype=wp.dtype_from_numpy(self.float_dtype))
        
            case 'numpy':
                return np.full((num_outputs,self.num_nodes),initial_value,dtype=self.float_dtype)
            case _:
                raise ValueError()
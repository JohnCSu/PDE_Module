import warp as wp
import numpy as np
import numba as nb
from pde_module.stencil import Stencil
from pde_module.mesh import Mesh
from pde_module.mesh.cell_types import CELLTYPES_DICT,CELLTYPE_DIMENSION_DICT
from pde_module.stencil.hooks import *
from typing import Optional

class ExplicitFiniteVolume(Stencil):
    def __init__(self, *args, **kwargs):
        super().__init__()
        



@nb.njit
def check_all_dim(dimension,cell_types,dim_dict):
    for cell_type in cell_types:
        if dim_dict[cell_type] != dimension:
            return False
    return True


class FiniteVolumeMesh(Mesh):
    '''
    We only support 3D cells
    '''
    cell_neighbors:np.ndarray
    faces:np.ndarray
    internal_faces:np.ndarray
    external_faces:np.ndarray
    cell_to_face:np.ndarray
    face_to_cell:np.ndarray
    face_areas:np.ndarray
    face_centroid:np.ndarray
    face_normal:np.ndarray
    backend:str = 'numpy'
    def __init__(self, nodes, cells_connectivity, cell_types, dimension = None, float_dtype=np.float32, int_dtype=np.int32):
        assert check_all_dim(3,cell_types,CELLTYPE_DIMENSION_DICT), 'Dimension of all celltypes must be 3D (Hex,Tetra,Wedge)'
        super().__init__(nodes, cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
        





class Fluxes(ExplicitFiniteVolume):
    mesh: Optional[FiniteVolumeMesh]
    interpolation: wp.Function
    kernel: wp.Kernel
    def __init__(self,interpolation = 'upwind',mesh:FiniteVolumeMesh = None):
        self.mesh = mesh
        self._face_to_cell = None
        super().__init__()
    def __call__(self,face_value,cell_values,face_to_cell = None):
        return super().__call__(face_value,cell_values,face_to_cell = None)
    
    @property
    def face_to_cell(self):
        return self._face_to_cell
    
    @setup
    def initialise(self,face_value,cell_values,face_to_cell = None):
        if face_to_cell is not None:
            self._face_to_cell = face_to_cell
        else:
            assert isinstance(self.mesh,FiniteVolumeMesh) , 'If face_to_cell not passed in then a Finite Volume mesh must be set during initialization'
        self.output_array = self.create_output_array(face_value)
        self.kernel
        
    def forward(self, face_value,cell_values):
        pass
        
            
            

@wp.func
def upwind(
    ownerVal,
    neighborVal,
):
    return

def create_fluxes_kernel(interpolation_func:wp.Function):
    @wp.kernel
    def calculate_fluxes(face_value,cell_values,face_to_cell,):
        varID,id = wp.tid()
        
        ownerCellID,neighborCellID = face_to_cell[id,0],face_to_cell[id,1]
        
        ownerValue, neighborValue = cell_values[id,0]
        
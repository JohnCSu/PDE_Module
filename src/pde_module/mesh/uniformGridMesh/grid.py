import numpy as np
from pde_module.mesh.mesh import Mesh
from math import prod
from pde_module.mesh.uniformGridMesh.functions import *


import numpy as np
from numba import njit, prange

class UniformGridMesh(Mesh):
    dx:float
    area:float
    volume:float
    ghost_cells:int
    nodal_grid: np.ndarray
    dimension: int
    origin : np.ndarray
    nodes_per_axis:tuple[int]
    num_cells: np.ndarray
    meshgrid:list[np.ndarray]
    coordinate_vectors:np.ndarray
    def __init__(self,dx,nodes_per_axis:tuple[int],origin = None,ghost_cells = None,float_dtype = np.float32,int_dtype = np.int32):
        dx = float_dtype(dx)
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.ghost_cells = get_ghost_cells(ghost_cells)
        
        self.origin = np.zeros(3,dtype=float_dtype) if origin is None else np.array(origin,dtype=float_dtype)
        assert len(self.origin) == 3,'Origin must be a tuple of length 3'
        
        self.nodes_per_axis,self.coordinate_vectors,self.meshgrid,self.nodal_grid = create_nodes_grid(dx,nodes_per_axis,self.origin,self.ghost_cells)
        
        self.num_cells = prod(i-1 for i in self.nodes_per_axis if i > 1)
        dimension = sum(1 for i in self.nodes_per_axis if i > 1)
        assert 1 <= dimension <= 3
        
        cells_connectivity,cell_types = cell_connectivity_and_type(self.nodes_per_axis,self.num_cells,dimension,int_dtype)
        
        super().__init__(self.nodal_grid.reshape(-1,3), cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
    
    
if __name__ == '__main__':
    import pyvista as pv
    mesh = UniformGridMesh(1,(3,2,2),ghost_cells=1)
    pv_mesh = pv.UnstructuredGrid(mesh.cells.connectivity,mesh.cells.types,mesh.nodes)
    pv_mesh.plot(show_edges=True)
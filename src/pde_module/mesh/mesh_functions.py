import pyvista as pv
import numpy as np
from .mesh import Mesh
from typing import Optional,Callable

def to_pyvista(mesh:Mesh):
    return pv.UnstructuredGrid(mesh.cells.connectivity,mesh.cells.types,mesh.nodes)

def create_field_from_mesh(mesh:Mesh,type:str,num_outputs:int,SoA = False,array_converter_func:Optional[Callable] = None):
    match type:
        case 'node':
            length = len(mesh.nodes)
        case 'cell':
            length = len(mesh.cells)
        case 'face':
            length = len(mesh.faces)
        case 'edge':
            length = len(mesh.edges)
        case _:
            raise ValueError('Valid type are: node,cell,face,edge')
        
    shape =(num_outputs,length) if SoA else (length,num_outputs) 
    arr = np.zeros(shape,dtype=mesh.float_dtype)
    if array_converter_func is not None:
        return array_converter_func(arr)
    return arr
    
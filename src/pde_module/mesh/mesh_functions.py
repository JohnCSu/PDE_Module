import pyvista as pv
import numpy as np
from .mesh import Mesh
from typing import Optional,Callable
from .uniformGridMesh import UniformGridMesh 
from warp.types import vector,matrix
import warp as wp

def to_pyvista(mesh:Mesh):
    '''
    Convert Mesh object to pyvista unstructured grid
    '''
    return pv.UnstructuredGrid(mesh.cells.connectivity,mesh.cells.types,mesh.nodes)

def create_field(mesh:Mesh,type:str,num_outputs:int,SoA = False,array_converter_func:Optional[Callable] = None,**kwargs):
    assert isinstance(mesh,Mesh)
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
        return array_converter_func(arr,**kwargs)
    return arr

def create_structured_field(mesh:UniformGridMesh,type:str,num_outputs:int,SoA = False,array_func:Optional[Callable] = None,**kwargs):
    assert isinstance(mesh,UniformGridMesh)
    match type:
        case 'node':
            shape = mesh.nodal_grid.shape[0:-1] # We dont need the last axis
        case 'cell':
            shape = tuple(n-1 for n in mesh.nodal_grid.shape[:-1])
        case _:
            raise ValueError('Valid types are: node andcell')
        
    shape =(num_outputs,*shape) if SoA else (*shape,num_outputs) 
    
    if array_func is not None:
        return array_func(shape = shape,**kwargs)
    else:
        return np.zeros(shape,dtype=mesh.float_dtype)


def create_structured_warp_field(mesh:UniformGridMesh,type:str,num_outputs:int | tuple[int],func:Optional[Callable] = None,**kwargs):
    assert isinstance(mesh,UniformGridMesh)
    match type:
        case 'node':
            shape = mesh.nodal_grid.shape[0:-1] # We dont need the last axis
        case 'cell':
            shape = tuple(n-1 for n in mesh.nodal_grid.shape[:-1])
        case _:
            raise ValueError('Valid types are: node andcell')
    
    if isinstance(num_outputs,int):
        dtype = vector(num_outputs,wp.dtype_from_numpy(mesh.float_dtype))
        # return wp.zeros(shape,dtype=)
    elif isinstance(num_outputs,tuple):
        dtype = matrix(num_outputs,wp.dtype_from_numpy(mesh.float_dtype))
        assert len(num_outputs) == 2
        # return wp.zeros(shape,dtype=))

    if func is not None:
        assert callable(func)
        arr = func(*mesh.meshgrid,**kwargs)    
        return wp.array(data = arr,shape = shape, dtype=dtype)
    else:
        return wp.zeros(shape,dtype=dtype)
    
    
    

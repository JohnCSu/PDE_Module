import pyvista as pv
import numpy as np
from .mesh import Mesh
from typing import Optional, Callable
from .uniformGridMesh import UniformGridMesh
from warp.types import vector, matrix
from pde_module.utils.types import wp_Array
import warp as wp


def to_pyvista(mesh: Mesh) -> pv.UnstructuredGrid:
    """Convert a Mesh object to a PyVista unstructured grid.

    Args:
        mesh: The Mesh object to convert.

    Returns:
        PyVista UnstructuredGrid ready for visualization.
    """
    return pv.UnstructuredGrid(mesh.cells.connectivity, mesh.cells.types, mesh.nodes)


def create_field(
    mesh: Mesh,
    type: str,
    num_outputs: int,
    SoA: bool = False,
    array_converter_func: Optional[Callable] = None,
    **kwargs,
) -> np.ndarray:
    """Create a numpy field array on a mesh.

    Args:
        mesh: The mesh to create the field on.
        type: Type of field ('node', 'cell', 'face', or 'edge').
        num_outputs: Number of output values per entity.
        SoA: If True, shape is (num_outputs, length), else (length, num_outputs).
        array_converter_func: Optional function to convert the array.
        **kwargs: Additional arguments to array_converter_func.

    Returns:
        Numpy array for the field.

    Raises:
        ValueError: If type is not valid.
    """
    assert isinstance(mesh, Mesh)
    match type:
        case "node":
            length = len(mesh.nodes)
        case "cell":
            length = len(mesh.cells)
        case "face":
            length = len(mesh.faces)
        case "edge":
            length = len(mesh.edges)
        case _:
            raise ValueError("Valid type are: node,cell,face,edge")

    shape = (num_outputs, length) if SoA else (length, num_outputs)
    arr = np.zeros(shape, dtype=mesh.float_dtype)
    if array_converter_func is not None:
        return array_converter_func(arr, **kwargs)
    return arr


def create_structured_field(
    mesh: UniformGridMesh,
    type: str,
    num_outputs: int,
    SoA: bool = False,
    array_func: Optional[Callable] = None,
    **kwargs,
) -> np.ndarray:
    """Create a numpy field array on a uniform grid mesh.

    Args:
        mesh: The UniformGridMesh to create the field on.
        type: Type of field ('node' or 'cell').
        num_outputs: Number of output values per entity.
        SoA: If True, shape is (num_outputs, ...), else (..., num_outputs).
        array_func: Optional function to create the array.
        **kwargs: Additional arguments to array_func.

    Returns:
        Numpy array for the field.

    Raises:
        ValueError: If type is not valid.
    """
    assert isinstance(mesh, UniformGridMesh)
    match type:
        case "node":
            shape = mesh.nodal_grid.shape[0:-1]
        case "cell":
            shape = tuple(n - 1 for n in mesh.nodal_grid.shape[:-1])
        case _:
            raise ValueError("Valid types are: node and cell")

    shape = (num_outputs, *shape) if SoA else (*shape, num_outputs)

    if array_func is not None:
        return array_func(shape=shape, **kwargs)
    else:
        return np.zeros(shape, dtype=mesh.float_dtype)


def create_structured_warp_field(
    mesh: UniformGridMesh,
    type: str,
    num_outputs: int | tuple[int, ...],
    AoS: bool = True,
    func: Optional[Callable] = None,
    **kwargs,
) -> wp_Array:
    """Create a Warp field array on a uniform grid mesh.

    Args:
        mesh: The UniformGridMesh to create the field on.
        type: Type of field ('node' or 'cell').
        num_outputs: Number of outputs (int for vector, tuple for matrix).
        func: Optional function to generate initial values from meshgrid.
        **kwargs: Additional arguments to func.

    Returns:
        Warp array for the field.

    Raises:
        ValueError: If type is not valid or num_outputs tuple is invalid.
    """
    assert isinstance(mesh, UniformGridMesh)
    
    
    match type:
        case "node":
            shape = mesh.nodal_grid.shape[0:-1] 
        case "cell":
            shape = tuple(n - 1 for n in mesh.nodal_grid.shape[:-1])
        case _:
            raise ValueError("Valid types are: node and cell")


    if AoS:
        if isinstance(num_outputs, int):
            dtype = vector(num_outputs, wp.dtype_from_numpy(mesh.float_dtype))
        elif isinstance(num_outputs, tuple):
            dtype = matrix(num_outputs, wp.dtype_from_numpy(mesh.float_dtype))
            assert len(num_outputs) == 2
    else:
        assert isinstance(num_outputs,int)
        dtype = wp.dtype_from_numpy(mesh.float_dtype)
        shape = (num_outputs,) + shape
    
    if func is not None:
        assert callable(func)
        arr = func(*mesh.meshgrid, **kwargs)
        return wp.array(data=arr, shape=shape, dtype=dtype)
    else:
        return wp.zeros(shape, dtype=dtype)

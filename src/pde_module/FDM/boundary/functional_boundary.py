'''
What we need:

For now we do a copy an then apply BC.

- Each point gets a flag value
- We need a function that checks if interior point -> Ez if 0 or Shape-1 then True sort of thing
- Edges use Second Order Von Neumann/Dirich, Interior Use First Order
- Have a function to set the values for 

'''
from .boundary_functions import get_constant_func
from pde_module.FDM.utils import eligible_dims_and_shift
import warp as wp
from warp.types import vector, matrix, type_is_vector
from pde_module.stencil.hooks import *
from pde_module.utils import get_unique_key,ijk_to_global_c,global_to_ijk_c
from pde_module.types import wp_Array, wp_Kernel
import numpy as np
from typing import Any, Callable
from math import prod
from pde_module.FDM.boundary.flags import DIRICHLET,VON_NEUMANN
from ..module import ExplicitUniformGridStencil
from dataclasses import dataclass
@dataclass
class FunctionBC:
    """Storage for function-based boundary condition data.

    Attributes:
        face_ids: Array of boundary face indices.
        boundary_type: Type of boundary condition (DIRICHLET or VON_NEUMANN).
        varID: Variable index this BC applies to.
        func: The function computing boundary values.
        kernel: Optional pre-compiled kernel.
    """

    face_ids: np.ndarray | wp.array
    boundary_type: np.int8 | wp.int8
    output_ids: int
    func: wp.Function
    kernel: wp.Kernel | None = None

    def to_warp(self) -> None:
        """Convert face_ids to warp array."""
        self.face_ids = wp.array(self.face_ids, dtype=int)
        self.boundary_type = self.boundary_type

    def create_kernel(self, input_dtype: vector, grid_shape,ghost_cells) -> None:
        """Create the boundary condition kernel.

        Args:
            input_dtype: The warp vector dtype for the field.
            dx: Grid spacing.
        """
        self.kernel = create_boundary_kernel(self.boundary_type,input_dtype, self.output_ids,grid_shape,ghost_cells,self.func,False)

class GridBoundary(ExplicitUniformGridStencil):
    """Apply boundary conditions around the perimeter of a uniform grid.

    Uses ghost cells to enforce boundary conditions with second-order accuracy.
    Currently only implemented for vector arrays.

    Args:
        field: Array that the boundary will apply to.
        dx: Grid spacing.
        ghost_cells: Number of ghost cells on the grid.
        grid_coordinates: Optional array of grid coordinates for function-based BCs.

    Boundary groups are accessed with strings: {-X,+X,-Y,+Y,-Z,+Z}.
    The string 'ALL' applies a BC to all boundary nodes.

    Supported BC:
        - Dirichlet
        - Von Neumann

    Convenience methods for vector fields matching dimension:
        - no_slip: Set all to 0.
        - impermeable: Normal velocity = 0.
    """

    def __init__(
        self,
        field: wp.array,
        dx: float,
        ghost_cells: int,
        grid_coordinates: np.ndarray | wp.array
    ) -> None:
        inputs = self.get_shape_from_dtype(field.dtype)
        self.dimension = sum(1 for g in field.shape if g > 1)
        
        
        
        super().__init__(inputs,inputs, dx, ghost_cells)

        if isinstance(grid_coordinates, np.ndarray):
            self.grid_coordinates = wp.array(
                grid_coordinates,
                dtype=vector(3, self.float_dtype),
            )
            
        elif wp.types.is_array(grid_coordinates):
            assert type_is_vector(grid_coordinates.dtype)
            self.grid_coordinates = grid_coordinates
        
        
        self.grid_shape = field.shape
        self.nodeIDs = np.arange(0,prod(self.grid_shape),1,dtype = np.int32)
        self.groups = {}
        self.func_groups:dict[str,FunctionBC] = {}
        
        self.define_exterior_groups()
        
    def define_exterior_groups(self):
        ghost_cells = self.ghost_cells
        
        slice_ghost = slice(ghost_cells,-(ghost_cells+1),1) if ghost_cells > 0 else slice(None)
        
        dims,_ = eligible_dims_and_shift(self.grid_shape,ghost_cells)
         
         
        self.slice_groups = {}
        
        for i,axis in enumerate(['X','Y','Z']):
            if self.grid_shape[i] > 1: 
                for prefix,sign in zip(['-','+'],[-1,1]):
                    key = prefix+axis
                    slice_group = [slice(None),slice(None),slice(None)]
                    
                    if sign == -1:
                        slice_group[i] = ghost_cells
                    else:
                        slice_group[i] = -(ghost_cells+1)

                    for x,g in enumerate(self.grid_shape):
                        if g > 1 and i != x:
                            slice_group[x] =  slice(ghost_cells,-(ghost_cells),1) if ghost_cells > 0 else slice(None)

                    self.slice_groups[key] = tuple(slice_group)
        
        temp_IDs = self.nodeIDs.reshape(self.grid_shape,copy = False)
        
        all_exterior_ls = []
        for i,(key,slice_idx) in enumerate(self.slice_groups.items()):
            self.groups[key] = temp_IDs[slice_idx].ravel()
            all_exterior_ls.append(self.groups[key])

        self.groups['ALL'] = np.unique(np.concatenate(all_exterior_ls,axis = 0)).astype(np.int32)
        return None
    
    def set_BC(self,
        face_ids: str | int | np.ndarray | list | tuple,
        value: float | Callable,
        boundary_type: int,
        output_ids: int = None,
    ) -> None:
        """Set a function-based boundary condition.

        Args:
            face_ids: Boundary face indices or group name.
            value: Float or warp.Function to compute boundary values.
            boundary_type: Type of boundary condition.
            outputs_id: Which output component to apply the BC to.
        """
        
        if output_ids is None:
            output_ids = tuple(i for i in range(self.inputs[0]))

        assert isinstance(output_ids,(int,list,tuple)), "output id must be an integer"
        groupName = None
        
        if not callable(value):
            assert isinstance(value, (float,int))
            value = get_constant_func(value,self.input_dtype,output_ids,self.float_dtype)
        
        assert isinstance(value,wp.Function), 'Function must be warp based'
        
        if isinstance(face_ids, str):
            assert face_ids in self.groups.keys()
            groupName = face_ids
            face_ids = self.groups[face_ids]
        else:
            assert isinstance(face_ids, (np.ndarray, list, tuple, int))
            groupName = get_unique_key(self.groups, "BC_Group")
            face_ids = np.array(face_ids, dtype=np.int32)
            assert len(face_ids.shape) == 1,'face_ids should be 1D'
        
        
        self.func_groups[groupName] = FunctionBC(
            face_ids, boundary_type, output_ids, value)
        


    def __call__(
        self,
        input_array: wp.array,
        t: float = 0.0,
        params: dict[str, Any] | None = None,
    ) -> wp_Array:
        """Apply boundary conditions to the input array.

        Args:
            input_array: Current values to apply BC to.
            t: Current simulation time (for function-based BCs).
            params: Parameters for function-based BC kernels.

        Returns:
            Array with boundary conditions applied.

        Note:
            A copy is performed between input and output before the forward call.
        """
        if params is None:
            params = {}
        return super().__call__(input_array, t, params)

    @setup
    def to_warp(self, *args, **kwargs) -> None:
        """Convert numpy arrays to warp arrays."""
        self.t = wp.zeros(1, dtype=self.float_dtype)
        
        for key in self.func_groups.keys():
            self.func_groups[key].to_warp()

    @setup(order=1)
    def initialize_kernel(self,input_array, *args, **kwargs) -> None:
        """Initialize the boundary kernel."""
        for key in self.func_groups.keys():
            self.func_groups[key].create_kernel(self.input_dtype, self.grid_shape,self.ghost_cells)

        self.output_array = self.create_output_array(input_array)
    @before_forward
    def set_default_params(
        self, input_array: wp.array, t: float, params: dict[str, Any], **kwargs
    ) -> None:
        """Set default parameters for function-based BCs."""
        for key in self.func_groups.keys():
            if key not in params.keys():
                params[key] = wp.uint8(0)

        self.t.fill_(self.float_dtype(t))

    @before_forward
    def copy_array(self, input_array: wp.array, *args, **kwargs) -> None:
        """Copy input array to output array before BC application."""
        wp.copy(self.output_array, input_array)

    def forward(
        self, input_array: wp.array, t: float, params: dict[str, Any], **kwargs
    ) -> wp_Array:
        """Apply boundary conditions.

        Args:
            input_array: Array to apply BC to.
            dt: Time step (used for function-based BCs).
            params: Parameters for function-based BCs.

        Returns:
            Array with boundary conditions applied.
        """
        for key, func_BC in self.func_groups.items():
            wp.launch(
                kernel=func_BC.kernel,
                dim=len(func_BC.face_ids),
                inputs=[
                    input_array,
                    func_BC.face_ids,
                    self.grid_coordinates,
                    self.t,
                    self.dx,
                    params[key],
                ],
                outputs=[self.output_array],
            )
            

        return self.output_array


def get_ids_to_set(ids):
    match ids:
        case int():
            length = 1
            ids = [ids]
        case list() | tuple():
            length = len(ids)
        case _:
            raise TypeError()
        
    ids_vector = vector(length,dtype = int)
    ids_to_set =ids_vector(*ids)
    return ids_to_set


def create_boundary_kernel(boundary_type,input_dtype: vector,output_ids,grid_shape, ghost_cells: int,function,is_ibm:bool) -> wp_Kernel:
    """Create a kernel for applying boundary conditions.

    Args:
        input_dtype: Warp vector dtype for the field.
        ghost_cells: Number of ghost cells.
        dx: Grid spacing.

    Returns:
        A wp.kernel for applying boundary conditions.
    """
    float_type = input_dtype._wp_scalar_type_

    dims,dims_shift = eligible_dims_and_shift(grid_shape,ghost_cells)
    dimension = len(dims)
    grid_shape = wp.vec3i(grid_shape)
    
    ids_to_set = get_ids_to_set(output_ids)
    NUM_VARS = len(ids_to_set)
    
    @wp.func
    def is_interior(nodeID:wp.vec3i):
        
        out_bool = True
        for ii in range(dimension):
            axis = dims[ii]
            out_bool = wp.where(not (nodeID[axis] >= (0 + ghost_cells) and (nodeID[axis] <= grid_shape[axis]-1-ghost_cells)),False,out_bool)
        return out_bool
    
    
    @wp.func
    def get_interior_vec(nodeID:wp.vec3i):
        interior_vec = wp.vec3i()
        for ii in range(dimension):
            axis = dims[ii]
            if nodeID[axis] == ghost_cells:
                interior_vec[axis] = 1
            elif nodeID[axis] == grid_shape[axis]-1-ghost_cells:
                interior_vec[axis] = -1
            # Otherwise leave as 0
        return interior_vec
    
    
    signs = wp.vec2i(-1,1)
    @wp.kernel
    def function_kernel(
        current_values: wp.array3d(dtype=input_dtype),
        boundaryIDs:wp.array(dtype= int),
        coordinates:wp.array3d(dtype=vector(3, float_type)),
        t:wp.array(dtype= float_type),
        dx:float_type,
        params:Any,
        new_values: wp.array3d(dtype=input_dtype),
    ):
        tid = wp.tid()

        global_id = boundaryIDs[tid]
        i,j,k = global_to_ijk_c(global_id,grid_shape[0],grid_shape[1],grid_shape[2])
        nodeID = wp.vec3i(i,j,k)    
        
        
        # Find the axis directions that point into the interior
        interior_vec = get_interior_vec(nodeID)
        # wp.printf('NodeID: %d %d %d, Interior: %d %d %d\n',nodeID[0],nodeID[1],nodeID[2],interior_vec[0],interior_vec[1],interior_vec[2])
        
        val = function(current_values,nodeID,coordinates,t[0],dx,params)
        if wp.static(boundary_type == DIRICHLET):
            new_values[i, j, k] = val
        
        inc_vec = wp.vec3i()
        for ii in range(dimension):
            axis = dims[ii]
            if interior_vec[axis] != 0:        
                inc_vec[axis] = interior_vec[axis]
                ghostID = nodeID - inc_vec
                adjID = nodeID + inc_vec
                current_value =current_values[adjID[0], adjID[1], adjID[2]]
                
                for id in range(NUM_VARS):
                    var = ids_to_set[id]
                    # new_values[i, j, k][var] = val[var]
                    if wp.static(boundary_type == DIRICHLET):
                        new_value = (float_type(2.0) * val[var]- current_value[var])
                    elif wp.static(boundary_type == VON_NEUMANN):
                        new_value = -wp.sign(float_type(inc_vec[axis])) * float_type(2.0) * dx * val[var] + current_value[var]
                    
                    new_values[ghostID[0], ghostID[1], ghostID[2]][var] = new_value
                inc_vec[axis] = 0 # Reset to zero
                

    return function_kernel

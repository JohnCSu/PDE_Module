from ..ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_float, is_float
from ...utils.constants import Boundary_Types
from ...utils.utils import get_unique_key
from ...stencil.hooks import *
import numpy as np
from typing import Callable, Any
from .func_boundary import FunctionBC


class Boundary(ExplicitUniformGridStencil):
    """Base class for boundary condition methods.

    Args:
        field: The array that the boundary will apply to.
        dx: Grid spacing.
        ghost_cells: Number of ghost cells on the grid.

    If inheriting this class, implement how to generate the required indices
    (i,j,k) for the boundary, then call define_boundary_value_and_type_arrays().

    Boundary groups are accessed with strings: {-X,+X,-Y,+Y,-Z,+Z}.
    The string 'ALL' applies a BC to all boundary nodes.

    Supported BC types:
        - Dirichlet
        - Von Neumann

    For vector fields with length matching dimension:
        - no_slip: Set all components to 0.
        - impermeable: Set normal velocity to 0.
    """

    boundary_type: np.ndarray
    boundary_value: np.ndarray
    groups: dict[str, np.ndarray[int]] 
    func_groups: dict[str, FunctionBC]

    def __init__(self, field: wp.array, dx: float, ghost_cells: int) -> None:
        inputs = self.get_shape_from_dtype(field.dtype)
        super().__init__(inputs, inputs, dx, field.dtype._wp_scalar_type_)
        assert type_is_vector(self.input_dtype), "input must be vector type"
        assert type(ghost_cells) is int and ghost_cells > 0
        self.ghost_cells = ghost_cells
        self.grid_shape_without_ghost = self.grid_shape_with_no_ghost_cells(
            field.shape, ghost_cells
        )
        self.grid_shape = field.shape
        self.dimension = self.calculate_dimension_from_grid_shape(self.grid_shape)

        self.groups = {}
        self.func_groups = {}
    def define_boundary_value_and_type_arrays(self, indices: np.ndarray) -> None:
        """Create boundary_value and boundary_type arrays.

        Also sets the 'ALL' group to the provided indices.
        """
        shape = (len(indices),) + self.input_dtype_shape
        self.boundary_ids = np.arange(len(indices))
        self.boundary_value = np.zeros(
            shape, dtype=wp.dtype_to_numpy(self.input_scalar_type)
        )
        self.boundary_type = np.ones_like(self.boundary_value, dtype=np.int8)
        self.groups["ALL"] = self.boundary_ids

    def _check_output_ids(
        self, output_ids: int | np.ndarray | list | tuple | None
    ) -> int | np.ndarray | slice:
        """Validate and normalize output_ids argument."""
        if output_ids is None:
            return slice(None)

        if isinstance(output_ids, int):
            assert type_is_vector(self.input_dtype)
            assert 0 <= output_ids < self.inputs[0]
            return output_ids

        if isinstance(output_ids, (list, tuple, np.ndarray)):
            output_ids = np.array(output_ids, dtype=np.int32)
            assert np.all(0 <= output_ids < self.inputs)
            return output_ids

        raise TypeError(
            f"Valid Types are: int|np.ndarray|list|tuple|None got {type(output_ids)} instead"
        )

    def set_BC(
        self,
        face_ids: str | int | np.ndarray | list | tuple,
        value: float,
        boundary_type: int,
        outputs_ids: int | np.ndarray | list | tuple | None,
    ) -> None:
        """Set a boundary condition.

        Args:
            face_ids: Boundary face indices or group name.
            value: The boundary value to set.
            boundary_type: Type of boundary condition (see Boundary_Types).
            outputs_ids: Which output components to apply the BC to.
        """
        if isinstance(face_ids, str):
            assert face_ids in self.groups.keys()
            face_ids = self.groups[face_ids]
        else:
            assert isinstance(face_ids, (np.ndarray, list, tuple, int))
            face_ids = np.array(face_ids, dtype=np.int32)

        outputs_ids = self._check_output_ids(outputs_ids)
        self.boundary_type[face_ids, outputs_ids] = boundary_type
        assert isinstance(value, float)
        self.boundary_value[face_ids, outputs_ids] = value

    def set_func_BC(
        self,
        face_ids: str | int | np.ndarray | list | tuple,
        func: Callable,
        boundary_type: int,
        outputs_id: int,
    ) -> None:
        """Set a function-based boundary condition.

        Args:
            face_ids: Boundary face indices or group name.
            func: Function to compute boundary values.
            boundary_type: Type of boundary condition.
            outputs_id: Which output component to apply the BC to.
        """
        assert type(outputs_id) is int, "output id must be an integer"
        groupName = None
        if isinstance(face_ids, str):
            assert face_ids in self.groups.keys()
            groupName = face_ids
            face_ids = self.groups[face_ids]
        else:
            assert isinstance(face_ids, (np.ndarray, list, tuple, int))
            groupName = get_unique_key(self.groups, "BC_Group")
            face_ids = np.array(face_ids, dtype=np.int32)

        self.func_groups[groupName] = FunctionBC(
            face_ids, np.int8(boundary_type), outputs_id, func
        )

    def dirichlet_BC(
        self,
        group: str | int | np.ndarray | list | tuple,
        value: float | Callable,
        outputs_ids: int | np.ndarray | list | tuple | None = None,
    ) -> None:
        """Apply a Dirichlet boundary condition.

        Args:
            group: Boundary group name or indices.
            value: Constant value or callable function.
            outputs_ids: Which components to apply the BC to.
        """
        if isinstance(value, (float, int)):
            self.set_BC(group, float(value), Boundary_Types.DIRICHLET, outputs_ids)
        else:
            assert callable(value)
            self.set_func_BC(group, value, Boundary_Types.DIRICHLET, outputs_ids)

    def vonNeumann_BC(
        self,
        group: str | int | np.ndarray | list | tuple,
        value: float | Callable,
        outputs_ids: int | np.ndarray | list | tuple | None = None,
    ) -> None:
        """Apply a Von Neumann boundary condition.

        Args:
            group: Boundary group name or indices.
            value: Constant value or callable function.
            outputs_ids: Which components to apply the BC to.
        """
        self.set_BC(group, value, Boundary_Types.VON_NEUMANN, outputs_ids)

    def no_slip(self, group: str | int | np.ndarray | list | tuple) -> None:
        """Apply no-slip boundary condition (velocity set to 0).

        Valid only when input_dtype is a vector with length equal to dimension.
        """
        assert self.inputs[0] == self.dimension, (
            "Valid only when input_dtype is vector with same length equal to dimension of field"
        )
        self.dirichlet_BC(group, 0.0)

    def impermeable(self, group: str) -> None:
        """Apply impermeable boundary condition (normal velocity set to 0).

        For side walls, sets normal velocity to zero while applying
        zero-gradient condition to tangential components.
        """
        assert self.inputs[0] == self.dimension, (
            "Valid only when input_dtype is vector with same length equal to dimension of field"
        )

        assert group in self.groups.keys() and group in {
            "-X",
            "+X",
            "-Y",
            "+Y",
            "-Z",
            "+Z",
        }, "{'-X','+X','-Y','+Y','-Z','+Z'} are valid groups"
        axis_name = group[-1]
        indices = ["X", "Y", "Z"]
        axis = indices.index(axis_name)
        self.vonNeumann_BC(group, 0.0)
        self.dirichlet_BC(group, 0.0, axis)

    def check_boundary_types(self) -> bool:
        """Check if all boundary IDs have a boundary type assigned.

        Returns:
            False if any boundary_type entry is NO_BC, True otherwise.
        """
        if np.any(self.boundary_type == Boundary_Types.NO_BC):
            Warning("There are boundary ID that have not been given a value yet")
            return False
        return True

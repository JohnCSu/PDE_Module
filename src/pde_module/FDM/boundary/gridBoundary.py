from .boundary import Boundary, FunctionBC
import warp as wp
from warp.types import vector, matrix, type_is_vector
from ...stencil.hooks import *
from ...utils.constants import *
from ...utils.types import wp_Array, wp_Kernel
import numpy as np
from typing import Any, Callable


class GridBoundary(Boundary):
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
        grid_coordinates: np.ndarray | wp.array | None = None,
    ) -> None:
        super().__init__(field, dx, ghost_cells)

        if isinstance(grid_coordinates, np.ndarray):
            self.grid_coordinates = wp.array(
                grid_coordinates,
                dtype=vector(3, wp.dtype_from_numpy(grid_coordinates.dtype)),
            )
        elif wp.types.is_array(grid_coordinates):
            assert type_is_vector(grid_coordinates.dtype)
            self.grid_coordinates = grid_coordinates
        else:
            self.grid_coordinates = None

        self.define_boundary_ijk_indices()
        self.define_groups()
        self.define_interior_adjacency()
        self.define_boundary_value_and_type_arrays(self.boundary_ijk_indices)

    def define_boundary_ijk_indices(self, *args, **kwargs) -> None:
        """Define the (i,j,k) indices for all boundary points."""
        boundary_ijk_indices = []

        for i in range(3):
            if self.grid_shape_without_ghost[i] == 1:
                continue
            axis_limits = (0, self.grid_shape_without_ghost[i] - 1)
            for axis_lim in axis_limits:
                side = list(self.grid_shape_without_ghost)
                side[i] = 1

                indices = np.indices(side, dtype=np.int32)
                indices = np.moveaxis(indices, 0, -1).reshape(-1, 3)
                indices[:, i] = axis_lim

                for ax in range(3):
                    if self.grid_shape_without_ghost[ax] != 1:
                        indices[:, ax] += self.ghost_cells

                boundary_ijk_indices.append(indices)

        self.boundary_ijk_indices = np.unique(
            np.concat(boundary_ijk_indices, axis=0, dtype=np.int32), axis=0
        ).astype(np.int32)

    def define_groups(self, *args, **kwargs) -> None:
        """Define boundary groups based on axis and direction."""
        self.boundary_ids = np.arange(len(self.boundary_ijk_indices), dtype=np.int32)
        self.groups["ALL"] = self.boundary_ids

        for axis, axis_name in enumerate(["X", "Y", "Z"]):
            if self.grid_shape_without_ghost[axis] == 1:
                continue
            coords = self.grid_shape_without_ghost[axis]

            axis_limits = [0 + self.ghost_cells, coords - 1 + self.ghost_cells]

            for axis_limit, parity in zip(axis_limits, ["-", "+"]):
                name = parity + axis_name
                self.groups[name] = self.boundary_ids[
                    self.boundary_ijk_indices[:, axis] == axis_limit
                ]

    def define_interior_adjacency(self) -> None:
        """Define the interior neighbor direction for each boundary point."""
        self.boundary_interior = np.zeros(
            shape=(len(self.boundary_ijk_indices), 3), dtype=np.int32
        )
        for i, axis in enumerate(["X", "Y", "Z"]):
            if self.grid_shape_without_ghost[i] == 1:
                continue
            for parity in ["-", "+"]:
                sign = 1 if parity == "-" else -1
                key = parity + axis
                index = self.groups[key]
                self.boundary_interior[index, i] = sign

    def shift_group(self, group: str, shift: int, axis: int) -> None:
        """Shift a group's indices along an axis.

        Useful for absorption layers behind inlets.

        Args:
            group: Group name to shift.
            shift: Number of cells to shift.
            axis: Axis along which to shift.
        """
        assert axis < self.dimension
        assert group in self.groups.keys()
        self.boundary_ijk_indices[self.groups[group], axis] += shift

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
        self.warp_boundary_ijk_indices = wp.array(
            self.boundary_ijk_indices, dtype=wp.vec3i
        )
        self.warp_boundary_interior = wp.array(self.boundary_interior, dtype=wp.vec3i)
        self.warp_boundary_type = wp.array(self.boundary_type)
        self.warp_boundary_value = wp.array(self.boundary_value, dtype=self.input_dtype)
        self.t = wp.zeros(1, dtype=self.float_dtype)

        if self.func_groups: # Non empty Dict
            assert self.grid_coordinates is not None
        
        for key in self.func_groups.keys():
            self.func_groups[key].to_warp()

    @setup(order=1)
    def initialize_kernel(self, *args, **kwargs) -> None:
        """Initialize the boundary kernel."""
        self.kernel = create_boundary_kernel(
            self.input_dtype, self.ghost_cells, self.dx
        )
        if self.func_groups:
            for key in self.func_groups.keys():
                self.func_groups[key].create_kernel(self.input_dtype, self.dx)

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

    def func_kernel(
        self, input_array: wp.array, t: float, params: dict[str, Any], **kwargs
    ) -> None:
        """Launch function-based boundary condition kernels."""
        for key, func_BC in self.func_groups.items():
            wp.launch(
                kernel=func_BC.kernel,
                dim=len(func_BC.face_ids),
                inputs=[
                    input_array,
                    func_BC.face_ids,
                    self.warp_boundary_ijk_indices,
                    self.warp_boundary_interior,
                    self.grid_coordinates,
                    self.t,
                    params[key],
                ],
                outputs=[self.output_array],
            )

    def forward(
        self, input_array: wp.array, dt: float, params: dict[str, Any], **kwargs
    ) -> wp_Array:
        """Apply boundary conditions.

        Args:
            input_array: Array to apply BC to.
            dt: Time step (used for function-based BCs).
            params: Parameters for function-based BCs.

        Returns:
            Array with boundary conditions applied.
        """
        wp.launch(
            kernel=self.kernel,
            dim=(len(self.boundary_ids), *self.input_dtype_shape),
            inputs=[
                input_array,
                self.warp_boundary_ijk_indices,
                self.warp_boundary_value,
                self.warp_boundary_type,
                self.warp_boundary_interior,
            ],
            outputs=[self.output_array],
        )

        if self.func_groups:
            self.func_kernel(input_array, dt, params)
        return self.output_array


def create_boundary_kernel(
    input_dtype: vector, ghost_cells: int, dx: float
) -> wp_Kernel:
    """Create a kernel for applying boundary conditions.

    Args:
        input_dtype: Warp vector dtype for the field.
        ghost_cells: Number of ghost cells.
        dx: Grid spacing.

    Returns:
        A wp.kernel for applying boundary conditions.
    """
    DIRICHLET = wp.int8(1)
    VON_NEUMANN = wp.int8(2)

    float_type = input_dtype._wp_scalar_type_

    @wp.kernel
    def boundary_kernel(
        current_values: wp.array3d(dtype=input_dtype),
        boundary_ijk_indices: wp.array(dtype=wp.vec3i),
        boundary_value: wp.array(dtype=input_dtype),
        boundary_type: wp.array2d(dtype=wp.int8),
        boundary_interior: wp.array(dtype=wp.vec3i),
        new_values: wp.array3d(dtype=input_dtype),
    ):
        i, var = wp.tid()

        nodeID = boundary_ijk_indices[i]
        x = nodeID[0]
        y = nodeID[1]
        z = nodeID[2]

        interior_vec = boundary_interior[i]
        val = boundary_value[i][var]

        for axis in range(3):
            if interior_vec[axis] != 0:
                inc_vec = wp.vec3i()
                inc_vec[axis] = interior_vec[axis]
                ghostID = nodeID - inc_vec
                adjID = nodeID + inc_vec
                if boundary_type[i][var] == DIRICHLET:
                    new_values[x, y, z][var] = val
                    new_values[ghostID[0], ghostID[1], ghostID[2]][var] = (
                        type(dx)(2.0) * val
                        - current_values[adjID[0], adjID[1], adjID[2]][var]
                    )
                elif boundary_type[i][var] == VON_NEUMANN:
                    new_values[ghostID[0], ghostID[1], ghostID[2]][var] = (
                        -wp.sign(float_type(inc_vec[axis])) * float_type(2.0) * dx * val
                        + current_values[adjID[0], adjID[1], adjID[2]][var]
                    )

    return boundary_kernel

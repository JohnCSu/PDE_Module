from .boundary import Boundary
import numpy as np
import warp as wp
from warp.types import vector, matrix, type_is_vector, is_array
from ...stencil.hooks import *
from pde_module.stencil.utils import (
    create_stencil_op,
    eligible_dims_and_shift,
    create_tensor_divergence_op,
)
from pde_module.utils.types import wp_Array
from matplotlib import pyplot as plt
from typing import Callable


class ImmersedBoundary(Boundary):
    """Define solid regions inside the grid and apply boundary conditions at the interface.

    Currently only implemented for vector arrays using a staircase (first-order) approximation.

    Args:
        field: Array that the boundary will apply to.
        dx: Grid spacing.
        ghost_cells: Number of ghost cells.

    Usage:
        - Use from_bool_func() to define solid regions via a boolean function.
        - Call finalize() after defining geometry to enable BC application.
        - Only the 'ALL' group is available by default.

    Supported BC:
        - Dirichlet
        - Von Neumann
    """

    def __init__(self, field: wp.array, dx: float, ghost_cells: int) -> None:
        super().__init__(field, dx, ghost_cells)
        self.bitmask = np.zeros(field.shape, dtype=np.int8)

    def from_bool_func(self, fn: Callable, meshgrid: list[np.ndarray]) -> None:
        """Define solid regions using a boolean function.

        The function should return 0 for points outside the solid and 1 for points inside.
        Example: f(x, y, z) -> sqrt(x**2 + y**2) < R

        Args:
            fn: Boolean function that takes meshgrid coordinates.
            meshgrid: List of coordinate arrays matching grid shape.
        """
        assert meshgrid[0].shape == self.grid_shape
        bitmask = fn(*meshgrid)
        self.bitmask += bitmask.astype(np.int8)

    def finalize(self) -> None:
        """Calculate final boundaries from the bitmask.

        Must be called after from_bool_func() and before applying BCs.
        """
        self._find_solid_boundary()
        self._find_fluid_boundary()
        self._find_fluid_neighbors()
        self.define_boundary_value_and_type_arrays(self.solid_boundary)

    def show_bitmask(self) -> None:
        """Display the bitmask visualization.

        Currently only supports 2D grids.
        """
        assert self.dimension == 2, "Currently only displaying 2D bitmasks supported"
        fig, ax = plt.subplots()
        if is_array(self.bitmask):
            bitmask = self.bitmask.numpy()
        else:
            bitmask = np.array(self.bitmask)
        im = ax.imshow(
            bitmask.squeeze().T,
            origin="lower",
            animated=False,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title("BitMask")

        plt.show()

    def _find_solid_boundary(self) -> None:
        """Find solid boundary indices from the bitmask."""
        self.solid_indices = np.stack(np.nonzero(self.bitmask), axis=-1, dtype=np.int32)
        is_solid_boundary = np.zeros(shape=len(self.solid_indices), dtype=np.bool_)

        locate_boundary_kernel = locate_boundary(self.grid_shape, self.ghost_cells)
        wp.launch(
            locate_boundary_kernel,
            len(self.solid_indices),
            inputs=[self.bitmask, self.solid_indices],
            outputs=[is_solid_boundary],
            device="cpu",
        )
        self.solid_boundary = self.solid_indices[is_solid_boundary]
        self.interior_solids = self.solid_indices[~is_solid_boundary]

        L, H, W = self.grid_shape
        x, y, z = (
            self.interior_solids[:, 0],
            self.interior_solids[:, 1],
            self.interior_solids[:, 2],
        )
        self.flat_array = (x * H * W) + y * W + z

    def _find_fluid_boundary(self) -> None:
        """Find fluid boundary indices adjacent to solid regions."""
        identify_fluid_boundary = create_identify_fluid_boundary_kernel(
            self.grid_shape, self.ghost_cells
        )
        max_neighbors = self.dimension * 2

        fluid_boundary = wp.empty(
            (len(self.solid_boundary), max_neighbors), dtype=wp.vec3i, device="cpu"
        )
        fluid_boundary.fill_(wp.vec3i(-1, -1, -1))

        wp.launch(
            identify_fluid_boundary,
            dim=len(self.solid_boundary),
            inputs=[self.bitmask, self.solid_boundary],
            outputs=[fluid_boundary],
            device="cpu",
        )
        fluid_boundary = fluid_boundary.flatten().numpy()

        self.fluid_boundary = np.unique(
            fluid_boundary[fluid_boundary[:, 0] != -1], axis=0
        )

    def _find_fluid_neighbors(self) -> None:
        """Identify fluid neighbors for each solid boundary point."""
        dim = self.dimension
        self.fluid_neighbors = np.zeros(
            shape=(len(self.solid_boundary), dim, 2), dtype=np.int8
        )
        identify_neighbors_kernel = identify_fluid_neighbors(
            self.grid_shape, self.ghost_cells
        )
        wp.launch(
            identify_neighbors_kernel,
            dim=len(self.solid_boundary),
            inputs=[self.bitmask, self.solid_boundary],
            outputs=[self.fluid_neighbors],
            device="cpu",
        )

    @setup
    def to_warp(self, *args, **kwargs) -> None:
        """Convert numpy arrays to warp arrays."""
        adj_matrix = matrix((self.dimension, 2), dtype=wp.int8)
        self.warp_solid_boundary_indices = wp.array(self.solid_boundary, dtype=wp.vec3i)
        self.warp_fluid_neighbors = wp.array(self.fluid_neighbors, dtype=adj_matrix)
        self.warp_boundary_type = wp.array(self.boundary_type)
        self.warp_boundary_value = wp.array(self.boundary_value, dtype=self.input_dtype)

        self.warp_interior_solids = wp.array(self.interior_solids, dtype=wp.vec3i)
        self.warp_interior_solid_indices = [
            wp.array(arr, dtype=int) for arr in np.moveaxis(self.interior_solids, 0, -1)
        ]

    @setup
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the boundary kernels."""
        self.kernel = create_staircase_boundary_kernel(
            input_array.dtype, self.grid_shape, self.ghost_cells, self.dx
        )
        self.fill_solids_inplace_kernel = create_fill_solids_inplace(input_array.dtype)

    @before_forward
    def copy_array(self, input_array: wp.array, *args, **kwargs) -> None:
        """Copy input array to output array."""
        wp.copy(self.output_array, input_array)

    def forward(
        self, input_array: wp.array, fill_value: float = 0.0, *args, **kwargs
    ) -> wp_Array:
        """Apply immersed boundary conditions.

        Args:
            input_array: Array to apply BC to.
            fill_value: Value to fill interior solid cells with.

        Returns:
            Array with boundary conditions applied.
        """
        wp.launch(
            self.kernel,
            dim=(len(self.warp_solid_boundary_indices), self.inputs[0]),
            inputs=[
                input_array,
                self.warp_solid_boundary_indices,
                self.warp_boundary_value,
                self.warp_boundary_type,
                self.warp_fluid_neighbors,
            ],
            outputs=[self.output_array],
        )

        wp.launch(
            self.fill_solids_inplace_kernel,
            dim=len(self.warp_interior_solids),
            inputs=[self.warp_interior_solids, fill_value],
            outputs=[self.output_array],
        )
        return self.output_array


def create_fill_solids_inplace(input_dtype: vector):
    """Create a kernel to fill interior solid cells with a value.

    Args:
        input_dtype: Warp vector dtype.

    Returns:
        A wp.kernel for filling solids.
    """

    @wp.kernel
    def fill_solids_inplace(
        solid_boundary_indices: wp.array(dtype=wp.vec3i),
        value: float,
        new_values: wp.array3d(dtype=input_dtype),
    ):
        tid = wp.tid()
        solidID = solid_boundary_indices[tid]
        x = solidID[0]
        y = solidID[1]
        z = solidID[2]
        new_values[x, y, z] = input_dtype(value)

    return fill_solids_inplace


def create_staircase_boundary_kernel(
    input_dtype: vector,
    grid_shape: tuple[int, ...],
    ghost_cells: int,
    dx: float,
):
    """Create a staircase boundary condition kernel.

    For staircase approximation:
        - Dirichlet: Set solid boundary to the value.
        - Von Neumann: Average from fluid neighbors.

    Args:
        input_dtype: Warp vector dtype.
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.
        dx: Grid spacing.

    Returns:
        A wp.kernel implementing the boundary condition.
    """
    DIRICHLET = wp.int8(1)
    VON_NEUMANN = wp.int8(2)

    FLUID_CELL = wp.int8(0)
    SOLID_CELL = wp.int8(1)

    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)

    dim = len(eligible_dims)
    adj_matrix = matrix((dim, 2), dtype=wp.int8)

    float_type = input_dtype._wp_scalar_type_
    dx = float_type(dx)
    shift = wp.vec2i(-1, 1)

    @wp.kernel
    def boundary_kernel(
        current_values: wp.array3d(dtype=input_dtype),
        solid_boundary_indices: wp.array(dtype=wp.vec3i),
        boundary_value: wp.array(dtype=input_dtype),
        boundary_type: wp.array2d(dtype=wp.int8),
        neighbors: wp.array(dtype=adj_matrix),
        new_values: wp.array3d(dtype=input_dtype),
    ):
        tid, var = wp.tid()

        solidID = solid_boundary_indices[tid]
        x = solidID[0]
        y = solidID[1]
        z = solidID[2]
        BC_val = boundary_value[tid][var]
        if boundary_type[tid][var] == DIRICHLET:
            new_values[x, y, z][var] = BC_val
        elif boundary_type[tid][var] == VON_NEUMANN:
            neighbor_mat = neighbors[tid]
            avg_value = float_type(0.0)
            n = float_type(1.0)
            for axis in range(dim):
                j = eligible_dims[axis]
                adj_vec = wp.vec3i()
                for i in range(2):
                    if neighbor_mat[axis, i] == FLUID_CELL:
                        adj_vec[j] = shift[i]

                        fluid_idx = solidID + adj_vec
                        contribution_val = (
                            current_values[fluid_idx[0], fluid_idx[1], fluid_idx[2]][
                                var
                            ]
                            - float_type(shift[i]) * BC_val * dx
                        )
                        avg_value += (contribution_val - avg_value) / n
                        n += float_type(1.0)

            new_values[x, y, z][var] = avg_value

    return boundary_kernel


def create_identify_fluid_boundary_kernel(
    grid_shape: tuple[int, ...], ghost_cells: int
):
    """Create a kernel to identify fluid boundary points adjacent to solids.

    Args:
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel for identifying fluid boundaries.
    """
    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)

    dim = len(eligible_dims)

    shift = wp.vec2i(-1, 1)

    @wp.kernel
    def identify_fluid_boundary(
        bit_array: wp.array3d(dtype=wp.int8),
        solid_indices: wp.array(dtype=wp.vec3i),
        fluid_indices: wp.array2d(dtype=wp.vec3i),
    ):
        tid = wp.tid()

        boundary_idx = solid_indices[tid]

        for d in range(dim):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            for i in range(2):
                adj_vec[j] = shift[i]
                adj = boundary_idx + adj_vec
                if bit_array[adj[0], adj[1], adj[2]] == 0:
                    fluid_indices[tid, i + j * 2] = adj

    return identify_fluid_boundary


def identify_fluid_neighbors(grid_shape: tuple[int, ...], ghost_cells: int):
    """Create a kernel to identify fluid neighbors for solid boundary points.

    Args:
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel for identifying neighbors.
    """
    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)

    dim = len(eligible_dims)
    adj_matrix = matrix((dim, 2), dtype=wp.int8)

    @wp.kernel
    def identify_neighbors_kernel(
        bit_array: wp.array3d(dtype=wp.int8),
        solid_indices: wp.array(dtype=wp.vec3i),
        adjacent: wp.array(dtype=adj_matrix),
    ):
        tid = wp.tid()

        boundary_idx = solid_indices[tid]
        mat = adj_matrix()

        for d in range(wp.static(len(eligible_dims))):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            adj_vec[j] = 1

            adj_l = boundary_idx - adj_vec
            adj_r = boundary_idx + adj_vec

            mat[d, 0] = bit_array[adj_l[0], adj_l[1], adj_l[2]]
            mat[d, 1] = bit_array[adj_r[0], adj_r[1], adj_r[2]]

        adjacent[tid] = mat

    return identify_neighbors_kernel


def locate_boundary(grid_shape: tuple[int, ...], ghost_cells: int):
    """Create a kernel to locate solid boundary points.

    Args:
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel for locating boundaries.
    """
    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)

    @wp.kernel
    def locate_boundary_kernel(
        bit_array: wp.array3d(dtype=wp.int8),
        solid_indices: wp.array(dtype=wp.vec3i),
        is_boundary_indices: wp.array(dtype=wp.bool),
    ):
        tid = wp.tid()

        solid_index = solid_indices[tid]

        for d in range(wp.static(len(eligible_dims))):
            j = eligible_dims[d]
            adj_vec = wp.vec3i()
            adj_vec[j] = 1

            adj_l = solid_index - adj_vec
            adj_r = solid_index + adj_vec

            if (bit_array[adj_l[0], adj_l[1], adj_l[2]] == wp.int8(0)) or (
                bit_array[adj_r[0], adj_r[1], adj_r[2]] == wp.int8(0)
            ):
                is_boundary_indices[tid] = True
                return

    return locate_boundary_kernel

import warp as wp
import numpy as np
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from warp.types import vector, matrix
from ..stencil.hooks import *
from pde_module.stencil.utils import eligible_dims_and_shift
from pde_module.utils.constants import INT32_MAX
from pde_module.utils.dummy_types import wp_Array, wp_Kernel,wp_Vector,wp_Matrix
from collections.abc import Iterable


class ViscousDampingLayer(ExplicitUniformGridStencil):
    """Apply a damping layer proportional to the time derivative.

    Creates a damping zone around the outer grid layers that applies a
    damping force proportional to the time derivative of the field.
    This is useful for absorbing waves at boundaries.

    Args:
        inputs: Shape of the input:
            - If int, input is a vector with length equal to grid dimension.
            - If tuple of 2 ints, input is a matrix.
        num_layers: Thickness of the damping layer in cells.
        grid_shape: Grid shape (3-tuple).
        dx: Grid spacing.
        ghost_cells: Number of ghost cells.
        p: Polynomial order for damping strength. Default is 2.
        float_dtype: Float type. Defaults to wp.float32.
    """

    def __init__(
        self,
        inputs: int | tuple[int, ...],
        num_layers: int,
        grid_shape: tuple[int, ...],
        dx: float,
        ghost_cells: int,
        p: int = 2,
        float_dtype=wp.float32,
    ) -> None:
        super().__init__(inputs, inputs, dx, ghost_cells, float_dtype=float_dtype)
        self.grid_shape = grid_shape
        self.num_layers = num_layers
        self.p = self.float_dtype(p)
        self.groups = {}
        self.exclude_group = set()
        self.get_damping_grid_points(num_layers, grid_shape)

    def exclude(self, *groups: str) -> None:
        """Exclude groups from the damping zone (e.g., inlet).

        Args:
            groups: Group names to exclude.
        """
        for group in groups:
            assert group in self.groups.keys()
            self.exclude_group.add(group)

    def get_damping_grid_points(
        self, num_layers: int, grid_shape: tuple[int, ...]
    ) -> None:
        """Calculate and store indices for damping layer points."""
        self.indices = np.indices(grid_shape, dtype=np.int32)
        self.indices = np.moveaxis(self.indices, 0, -1).reshape(-1, 3)

        for i, (s, axis_name) in enumerate(zip(grid_shape, ["X", "Y", "Z"])):
            if s > 1:
                a1 = self.indices[:, i] < num_layers
                a2 = self.indices[:, i] >= (s - num_layers)

                self.groups[f"-{axis_name}"] = a1
                self.groups[f"+{axis_name}"] = a2

    def get_damping_array(self) -> np.ndarray:
        """Get the combined damping layer indices.

        Returns:
            Array of grid indices for the damping layer.
        """
        masks = [
            val for key, val in self.groups.items() if key not in self.exclude_group
        ]
        mask = masks[0]
        for m in masks[1:]:
            mask = np.logical_or(mask, m)

        outer_layers = self.indices[mask]
        return outer_layers

    @staticmethod
    def calculate_C_max(c: float, L: float, p: float = 2, R: float = 1e-2) -> float:
        """Calculate the maximum damping coefficient for stability.

        Args:
            c: Wave speed.
            L: Thickness of damping layer in distance units.
            p: Polynomial order.
            R: Target reflection coefficient.

        Returns:
            Maximum damping coefficient C_max.
        """
        return float((p + 1.0) * c / (2.0 * L) * np.log(1.0 / R))

    def __call__(self, du_dt: wp.array, C_max: float) -> wp_Array:
        """Apply the viscous damping layer.

        Args:
            du_dt: 3D array representing the time derivative.
            C_max: Damping coefficient. Recommended: (p+1)c/(2L) * ln(1/R).
                For stability: C_max * dt <= 0.5.

        Returns:
            3D array with same shape and dtype as input.
        """
        return super().__call__(du_dt, C_max)

    @setup
    def initialize_kernel(self, du_dt: wp.array, C_max: float) -> None:
        """Initialize the damping layer kernel."""
        self.outer_layers = self.get_damping_array()
        self.warp_outer_points = wp.array(self.outer_layers, dtype=wp.vec3i)
        self.kernel = create_dampingLayer_kernel(
            self.input_dtype, self.num_layers, self.p, self.grid_shape, self.ghost_cells
        )

    @setup
    def zero_array(self, *args, **kwargs) -> None:
        """Zero the output array."""
        self.output_array.zero_()

    def forward(self, du_dt: wp.array, C_max: float) -> wp_Array:
        """Apply the viscous damping.

        Args:
            du_dt: 3D array representing the time derivative.
            C_max: Scaling factor for the damping.

        Returns:
            3D array with same shape and dtype as input.
        """
        wp.launch(
            self.kernel,
            dim=len(self.warp_outer_points),
            inputs=[du_dt, self.warp_outer_points, C_max],
            outputs=[self.output_array],
        )
        return self.output_array


def create_dampingLayer_kernel(
    input_dtype: wp_Vector | wp_Matrix,
    num_layers: int,
    p: float,
    grid_shape: tuple[int, ...],
    ghost_cells: int,
) -> wp_Kernel:
    """Create a kernel for the viscous damping layer.

    Args:
        input_dtype: Warp vector or matrix dtype.
        num_layers: Thickness of damping layer.
        p: Polynomial order for damping profile.
        grid_shape: Shape of the grid.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel implementing the damping layer.
    """
    eligible_dims, _ = eligible_dims_and_shift(grid_shape, ghost_cells)
    dimension = len(eligible_dims)
    limits = matrix(shape=(3, 2), dtype=int)()

    for i, s in enumerate(grid_shape):
        if s > 1:
            limits[i] = wp.vec2i([num_layers, (s - 1) - num_layers])

    float_type = input_dtype._wp_scalar_type_
    grid_shape_vec = wp.vec3i(grid_shape)

    @wp.func
    def get_argmin_and_min(grid_point: wp.vec3i) -> wp.int32:
        out = wp.vec3i()

        for i in range(3):
            boundary = grid_shape_vec[i]
            point = grid_point[i]
            out[i] = wp.where(
                boundary > 1, wp.min(point, boundary - point - 1), INT32_MAX
            )
        return wp.int32(wp.argmin(out))

    @wp.kernel
    def dampingLayer(
        du_dt: wp.array3d(dtype=input_dtype),
        grid_points: wp.array(dtype=wp.vec3i),
        C_max: float_type,
        output_array: wp.array3d(dtype=input_dtype),
    ):
        tid = wp.tid()

        grid_point = grid_points[tid]

        axis = get_argmin_and_min(grid_point)

        assert 0 <= axis < 2

        x = grid_point[axis]
        if (grid_shape_vec[axis] - 1 - x) < x:
            dx = x - limits[axis, 1]
        else:
            dx = limits[axis, 0] - x

        damp_factor = C_max * (float_type(dx) / float_type(num_layers)) ** p
        output_array[grid_point[0], grid_point[1], grid_point[2]] = (
            -damp_factor * du_dt[grid_point[0], grid_point[1], grid_point[2]]
        )

    return dampingLayer

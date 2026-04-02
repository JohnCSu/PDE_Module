import warp as wp
import numpy as np
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from warp.types import vector, matrix
from ...stencil.hooks import *
from pde_module.utils.constants import INT32_MAX
from pde_module.utils.types import wp_Array, wp_Kernel, wp_Vector, wp_Matrix
from collections.abc import Iterable
from pde_module.FDM.Kernel import create_dampingLayer_kernel
from pde_module.FDM.Functional import viscousDampingLayer


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
        return viscousDampingLayer(
            self.kernel,
            du_dt,
            self.warp_outer_points,
            C_max,
            self.output_array,
            device=self.device,
        )

import warp as wp
import numpy as np
from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
from warp.types import vector, matrix
from ...stencil.hooks import *
from pde_module.utils.constants import INT32_MAX
from pde_module.utils.types import wp_Array, wp_Vector, wp_Matrix
from collections.abc import Iterable
from pde_module.FDM.kernel import create_spongeLayer_kernel
from pde_module.FDM.functional import farfield


class FarField(ExplicitUniformGridStencil):
    """Apply a sponge layer to absorb waves at the boundary.

    Creates a damping zone around the outer layers of the grid that gradually
    forces the solution to match a far-field condition. The output is:
        -sigma * (input_array[grid_point] - farfield_condition)

    Args:
        inputs: Shape of the input:
            - If int, input is a vector with length equal to grid dimension.
            - If tuple of 2 ints, input is a matrix.
        num_layers: Thickness of the sponge layer in cells.
        beta: Polynomial order for the damping profile.
        grid_shape: Grid shape (3-tuple).
        dx: Grid spacing.
        ghost_cells: Number of ghost cells.
        float_dtype: Float type. Defaults to wp.float32.
    """

    def __init__(
        self,
        inputs: int | tuple[int, ...],
        num_layers: int,
        beta: float,
        grid_shape: tuple[int, ...],
        dx: float,
        ghost_cells: int,
        float_dtype=wp.float32,
    ) -> None:
        super().__init__(inputs, inputs, dx, ghost_cells, float_dtype=float_dtype)
        self.grid_shape = grid_shape
        self.num_layers = num_layers
        self.beta = beta
        self.groups = {}
        self.exclude_group = set()
        self.get_sponge_sponge_points(num_layers, grid_shape)

    def exclude(self, *groups: str) -> None:
        """Exclude groups from the sponge zone (e.g., inlet).

        Args:
            groups: Group names to exclude from the sponge layer.
        """
        for group in groups:
            assert group in self.groups.keys()
            self.exclude_group.add(group)

    def get_sponge_sponge_points(
        self, num_layers: int, grid_shape: tuple[int, ...]
    ) -> None:
        """Calculate and store indices for sponge layer points."""
        self.indices = np.indices(grid_shape, dtype=np.int32)
        self.indices = np.moveaxis(self.indices, 0, -1).reshape(-1, 3)

        for i, (s, axis_name) in enumerate(zip(grid_shape, ["X", "Y", "Z"])):
            if s > 1:
                a1 = self.indices[:, i] < num_layers
                a2 = self.indices[:, i] >= (s - num_layers)

                self.groups[f"-{axis_name}"] = a1
                self.groups[f"+{axis_name}"] = a2

    def get_sponge_array(self) -> np.ndarray:
        """Get the combined sponge layer indices.

        Returns:
            Array of grid indices for the sponge layer.
        """
        masks = [
            val for key, val in self.groups.items() if key not in self.exclude_group
        ]
        mask = masks[0]
        for m in masks[1:]:
            mask = np.logical_or(mask, m)

        outer_layers = self.indices[mask]
        return outer_layers

    def __call__(
        self,
        input_array: wp.array,
        farfield_condition: wp_Vector | wp_Matrix,
        sigma_max: float,
    ) -> wp_Array:
        """Apply the far-field sponge layer boundary condition.

        Args:
            input_array: 3D array matching input shape.
            farfield_condition: Value that the solution is forced toward.
            sigma_max: Scaling factor. Recommended: alpha/dt where alpha ~ 0.05-0.5.

        Returns:
            3D array with same shape and dtype as input_array.
        """
        return super().__call__(input_array, farfield_condition, sigma_max)

    @setup
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the sponge layer kernel."""
        self.outer_layers = self.get_sponge_array()
        self.warp_outer_points = wp.array(self.outer_layers, dtype=wp.vec3i)
        self.kernel = create_spongeLayer_kernel(
            self.input_dtype,
            self.num_layers,
            self.beta,
            self.grid_shape,
            self.ghost_cells,
        )

    @setup
    def zero_array(self, *args, **kwargs) -> None:
        """Zero the output array."""
        self.output_array.zero_()

    def forward(
        self,
        input_array: wp.array,
        farfield_condition: wp_Vector | wp_Matrix,
        sigma_max: float,
    ) -> wp_Array:
        """Apply the sponge layer damping.

        Args:
            input_array: 3D array matching input shape.
            farfield_condition: Value that the solution is forced toward.
            sigma_max: Scaling factor for the damping.

        Returns:
            3D array with same shape and dtype as input_array.
        """
        return farfield(
            self.kernel,
            input_array,
            self.warp_outer_points,
            farfield_condition,
            sigma_max,
            self.output_array,
            device=self.device,
        )

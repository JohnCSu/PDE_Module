from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import types_equal
from ...stencil.hooks import *
from pde_module.utils.types import *
from pde_module.FDM.kernel import create_Laplacian_kernel
from pde_module.FDM.functional import laplacian


class Laplacian(ExplicitUniformGridStencil):
    """Compute the Laplacian of a vector field using central finite differences.

    This stencil calculates the Laplacian operator (div grad) for a vector field
    on a uniform grid using a second-order central difference scheme.

    Args:
        inputs: Length of the input vector.
        dx: Grid spacing.
        ghost_cells: Number of ghost cells on the grid.
        stencil: Optional custom stencil. If None, a 2nd order stencil is used.
        float_dtype: Float type for computations. Defaults to wp.float32.
    """

    def __init__(
        self,
        inputs: int,
        dx: float,
        ghost_cells: int,
        stencil: wp_Vector | None = None,
        float_dtype=wp.float32,
    ) -> None:
        if stencil is None:
            self.stencil = wp.types.vector(3, dtype=float_dtype)(
                [1.0 / dx**2, -2.0 / dx**2, 1.0 / dx**2]
            )
        else:
            raise ValueError("Custom stencil not implemented yet")
        assert (self.stencil._length_ % 2) == 1, "stencil must be odd sized"

        super().__init__(inputs, inputs, dx, ghost_cells, float_dtype=float_dtype)

    @setup
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the Laplacian kernel."""
        assert types_equal(self.input_dtype, input_array.dtype)
        assert len(self.inputs) == 1, "Laplacian Only For Vectors"
        self.kernel = create_Laplacian_kernel(
            self.input_dtype, input_array.shape, self.stencil, self.ghost_cells
        )
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(
            input_array.shape, self.ghost_cells
        )

    def forward(
        self, input_array: wp.array, alpha: float = 1.0, *args, **kwargs
    ) -> wp_Array:
        """Compute the Laplacian of the input field.

        Args:
            input_array: 3D array with vector dtype representing the field.
            alpha: Scaling factor for the Laplacian term. Default is 1.0.

        Returns:
            3D array with same vector dtype representing the Laplacian of each term.
        """
        return laplacian(
            self.kernel,
            input_array,
            alpha,
            self.output_array,
            self.ghost_cells,
            device=self.device,
        )

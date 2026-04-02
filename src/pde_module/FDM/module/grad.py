from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import types_equal
from ...stencil.hooks import *
from pde_module.utils.types import *
from pde_module.FDM.kernel import create_Grad_kernel
from pde_module.FDM.functional import grad


class Grad(ExplicitUniformGridStencil):
    """Compute the gradient of a field using central finite differences.

    Uses a second-order central difference scheme to compute the gradient.

    Args:
        inputs: Length of the input vector.
        grid_shape: Grid shape (3-tuple, with 1 indicating invalid dimension).
        dx: Grid spacing.
        ghost_cells: Number of ghost cells on the grid.
        stencil: Optional custom stencil. If None, 2nd Order stencil is used.
        force_matrix: If True and inputs == 1, output is a matrix instead of vector.
        float_dtype: Float type for computations. Defaults to wp.float32.
    """

    def __init__(
        self,
        inputs: int,
        grid_shape: tuple[int, ...],
        dx: float,
        ghost_cells: int,
        stencil: wp_Vector | None = None,
        force_matrix: bool = False,
        float_dtype=wp.float32,
    ) -> None:
        self.force_matrix = force_matrix
        dimension = self.calculate_dimension_from_grid_shape(grid_shape)
        self.stencil = wp.types.vector(3, dtype=float_dtype)(
            [-1.0 / (2 * dx), 0.0, 1.0 / (2 * dx)]
        )

        assert type(inputs) is int and inputs > 0

        if inputs == 1 and force_matrix is False:
            output_shape = dimension
        else:
            output_shape = (inputs, dimension)

        assert dimension <= 3

        super().__init__(inputs, output_shape, dx, ghost_cells, float_dtype)

    @setup
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the gradient kernel."""
        assert types_equal(self.input_dtype, input_array.dtype)
        assert len(self.inputs) == 1, "Grad Only For Vectors"
        self.kernel = create_Grad_kernel(
            self.input_dtype,
            self.output_dtype,
            input_array.shape,
            self.stencil,
            self.ghost_cells,
        )
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(
            input_array.shape, self.ghost_cells
        )

    def forward(self, input_array: wp.array, alpha: float = 1.0) -> wp_Array:
        """Compute the gradient of the input field.

        Args:
            input_array: 3D array with vector dtype to calculate gradient from.
            alpha: Scaling factor. Default is 1.0.

        Returns:
            3D array where each element is a matrix or vector:
            - If vector length (N) == 1: returns vector of dimension (D) of the grid,
              or (1, D) matrix if force_matrix=True.
            - If vector length (N) > 1: returns matrix of size (N, D).
        """
        return grad(
            self.kernel,
            input_array,
            alpha,
            self.output_array,
            self.ghost_cells,
            device=self.device,
        )

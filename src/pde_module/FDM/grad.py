from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from ..stencil.hooks import *
from pde_module.stencil.utils import create_stencil_op, eligible_dims_and_shift
from pde_module.utils.types import *

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
        wp.launch(
            self.kernel,
            dim=self.kernel_dim,
            inputs=[input_array, alpha],
            outputs=[self.output_array],
        )
        return self.output_array


def create_Grad_kernel(
    input_vector: wp_Vector,
    output_dtype: wp_Vector | wp_Matrix,
    grid_shape: tuple[int, ...],
    stencil: wp_Vector,
    ghost_cells: int,
):
    """Create a kernel for computing the gradient.

    Args:
        input_vector: Warp vector dtype for the input field.
        output_dtype: Warp vector or matrix dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the gradient.
    """
    assert type_is_vector(input_vector)

    output_dtype_is_vector = type_is_vector(output_dtype)
    assert type_is_vector(output_dtype) or type_is_matrix(output_dtype)

    stencil_op = create_stencil_op(input_vector, stencil, ghost_cells)
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    @wp.kernel
    def grad_kernel(
        scalar_array: wp.array3d(dtype=input_vector),
        alpha: input_vector._wp_scalar_type_,
        grad_array: wp.array3d(dtype=output_dtype),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        grad = output_dtype()

        for i in range(wp.static(len(dims))):
            if wp.static(output_dtype_is_vector):
                grad[i] = stencil_op(scalar_array, index, stencil, dims[i])[0]
            else:
                grad[:, i] = stencil_op(scalar_array, index, stencil, dims[i])

        grad *= alpha

        grad_array[index[0], index[1], index[2]] = grad

    return grad_kernel

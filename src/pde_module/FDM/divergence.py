from .ExplicitUniformGridStencil import ExplicitUniformGridStencil
import warp as wp
from warp.types import vector, matrix, type_is_vector, type_is_matrix, types_equal
from ..stencil.hooks import *
from pde_module.stencil.utils import (
    create_stencil_op,
    eligible_dims_and_shift,
    create_tensor_divergence_op,
)
from collections.abc import Iterable
from pde_module.utils.dummy_types import *

class Divergence(ExplicitUniformGridStencil):
    """Compute the divergence of a field using central finite differences.

    Uses a second-order central difference scheme.

    Args:
        inputs: Shape of the input:
            - If int, input is a vector with length equal to grid dimension.
            - If tuple of 2 ints, input is a matrix with columns equal to grid dimension.
        grid_shape: Grid shape (3-tuple, with 1 indicating invalid dimension).
        dx: Grid spacing.
        ghost_cells: Number of ghost cells on the grid.
        stencil: Optional custom stencil. If None, 2nd Order stencil is used.
        float_dtype: Float type for computations. Defaults to wp.float32.
    """

    def __init__(
        self,
        inputs: int | tuple[int, ...],
        grid_shape: tuple[int, ...],
        dx: float,
        ghost_cells: int,
        stencil: wp_Vector | None = None,
        float_dtype=wp.float32,
    ) -> None:
        if stencil is None:
            self.stencil = wp.types.vector(3, dtype=float_dtype)(
                [-1.0 / (2 * dx), 0.0, 1.0 / (2 * dx)]
            )
        else:
            raise ValueError("Custom stencil not implemented yet")
        assert (self.stencil._length_ % 2) == 1, "stencil must be odd sized"

        dimension = self.calculate_dimension_from_grid_shape(grid_shape)

        is_vector = False
        if isinstance(inputs, Iterable):
            assert all(type(i) is int for i in inputs)
            if len(inputs) == 1:
                inputs = inputs[0]
                is_vector = True
            else:
                assert len(inputs) == 2
        elif type(inputs) is int:
            is_vector = True
        else:
            raise ValueError("Inputs must be int or tuple|list of int")

        if is_vector:
            assert inputs == dimension
            outputs = 1
        else:
            assert inputs[1] == dimension
            outputs = inputs[0]

        super().__init__(inputs, outputs, dx, ghost_cells, float_dtype=float_dtype)

    @setup
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the divergence kernel."""
        assert types_equal(self.input_dtype, input_array.dtype)

        self.kernel = create_Divergence_kernel(
            self.input_dtype,
            self.output_dtype,
            input_array.shape,
            self.stencil,
            self.ghost_cells,
        )
        self.kernel_dim = self.grid_shape_with_no_ghost_cells(
            input_array.shape, self.ghost_cells
        )

    def forward(
        self, input_array: wp.array, alpha: float = 1.0, *args, **kwargs
    ) -> wp_Array:
        """Compute the divergence of the input field.

        Args:
            input_array: 3D array matching input shape (vector or matrix).
            alpha: Scaling factor. Default is 1.0.

        Returns:
            3D array of vectors:
            - If vector input: output is a vector of length 1 (scalar).
            - If matrix input of size (N, D): output is a vector of size (N,).
        """
        wp.launch(
            self.kernel,
            dim=self.kernel_dim,
            inputs=[input_array, alpha],
            outputs=[self.output_array],
        )
        return self.output_array


def create_Divergence_kernel(
    input_dtype: wp_Vector | wp_Matrix,
    output_vector: vector,
    grid_shape: tuple[int, ...],
    stencil: vector,
    ghost_cells: int,
):
    """Create a kernel for computing the divergence.

    Args:
        input_dtype: Warp vector or matrix dtype for the input field.
        output_vector: Warp vector dtype for the output.
        grid_shape: Shape of the grid (3-tuple).
        stencil: Vector containing stencil weights.
        ghost_cells: Number of ghost cells.

    Returns:
        A wp.kernel that computes the divergence.
    """
    dims, dims_shift = eligible_dims_and_shift(grid_shape, ghost_cells)

    if type_is_vector(input_dtype):
        div_op = create_stencil_op(input_dtype, stencil, ghost_cells)
    else:
        div_op = create_tensor_divergence_op(
            input_dtype, stencil, grid_shape, ghost_cells
        )

    @wp.kernel
    def Divergence_kernel(
        input_values: wp.array3d(dtype=input_dtype),
        alpha: input_dtype._wp_scalar_type_,
        output_values: wp.array3d(dtype=output_vector),
    ):
        i, j, k = wp.tid()

        index = wp.vec3i(i, j, k)
        index += dims_shift

        div = output_vector()
        if wp.static(type_is_matrix(input_dtype)):
            div = div_op(input_values, index, stencil)
        else:
            for i in range(wp.static(len(dims))):
                div[0] += div_op(input_values, index, stencil, dims[i])[i]

        div *= alpha

        output_values[index[0], index[1], index[2]] = div

    return Divergence_kernel

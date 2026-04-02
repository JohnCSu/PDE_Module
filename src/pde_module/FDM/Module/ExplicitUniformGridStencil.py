import warp as wp
from warp.types import vector, matrix
from ...utils import *
from ...stencil.stencil import Stencil
from ...stencil.hooks import *
from collections.abc import Iterable


class ExplicitUniformGridStencil(Stencil):
    """Base class for explicit stencil operations on uniform grids.

    Designed for array-of-structures (AoS) grids where each grid point
    contains a vector or matrix value.
    """

    output_array: wp.array | None = None

    def __init__(
        self,
        inputs: int | list[int],
        outputs: int | list[int],
        dx: float,
        ghost_cells: int = 0,
        float_dtype: wp.float32 | wp.float64 = wp.float32,
    ) -> None:
        super().__init__()
        self.dx = float_dtype(dx)
        self._inputs = tuplify(inputs)
        self._outputs = tuplify(outputs)
        self.ghost_cells = ghost_cells

        self.float_dtype = float_dtype
        self._input_dtype = self._get_dtype_from_shape(self.inputs, float_dtype)
        self._output_dtype = self._get_dtype_from_shape(self.outputs, float_dtype)

    @property
    def output_dtype(self) -> wp_Matrix | wp_Vector:
        """Dtype of the output array."""
        return self._output_dtype

    @property
    def input_dtype(self) -> wp_Matrix | wp_Vector:
        """Dtype of the input array."""
        return self._input_dtype

    @property
    def inputs(self) -> tuple[int, ...]:
        """Shape of input as a tuple of integers."""
        assert 1 <= len(self._outputs) <= 2, (
            "AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass"
        )
        return self._inputs

    @property
    def outputs(self) -> tuple[int, ...]:
        """Shape of output as a tuple of integers."""
        assert 1 <= len(self._outputs) <= 2, (
            "AoS stencils input/output can be either vec (length 1) or matrix (length 2) check input and output pass"
        )
        return self._outputs

    @property
    def input_dtype_shape(self) -> tuple[int, ...]:
        """Shape of input dtype as a tuple."""
        return self.get_shape_from_dtype(self.input_dtype)

    @property
    def output_dtype_shape(self) -> tuple[int, ...]:
        """Shape of output dtype as a tuple."""
        return self.get_shape_from_dtype(self.output_dtype)

    @property
    def output_scalar_type(self):
        """Warp scalar dtype (e.g. float32) of output dtype."""
        return self.output_dtype._wp_scalar_type_

    @property
    def input_scalar_type(self):
        """Warp scalar dtype (e.g. float32) of input dtype."""
        return self.output_dtype._wp_scalar_type_

    @staticmethod
    def calculate_dimension_from_grid_shape(shape: tuple[int, ...]) -> int:
        """Calculate the number of valid dimensions (where size > 1)."""
        return sum(1 for s in shape if s > 1)

    @staticmethod
    def _get_dtype_from_shape(
        shape: tuple[int, ...], float_dtype
    ) -> wp_Vector | wp_Matrix:
        """Get the warp dtype from a shape tuple."""
        return dtype_from_shape(shape, float_dtype)

    @staticmethod
    def get_shape_from_dtype(dtype) -> tuple[int, ...]:
        """Get the shape of a dtype as a tuple.

        Args:
            dtype: A warp vector or matrix dtype.

        Returns:
            Tuple representing the shape.

        Raises:
            TypeError: If dtype is not a warp vector or matrix.
        """
        if wp.types.type_is_vector(dtype):
            return (dtype._length_,)
        elif wp.types.type_is_matrix(dtype):
            return dtype._shape_
        else:
            raise TypeError("Dtypes supported are warp vector and matrix only")

    @staticmethod
    def get_ghost_shape_from_stencil(
        grid_shape: tuple[int, ...], stencil: vector
    ) -> tuple[int, ...]:
        """Calculate field shape accounting for ghost cells used by stencil.

        Args:
            grid_shape: Full grid shape including ghost cells.
            stencil: The stencil vector.

        Returns:
            Shape of the interior grid (excluding ghost cell region).
        """
        length = stencil._length_
        assert (length % 2) == 1, "stencil must be odd sized"
        num_ghost_cells = length - 1
        shape = tuple(s - num_ghost_cells for s in grid_shape if s > 1)
        return shape

    @staticmethod
    def grid_shape_with_no_ghost_cells(
        grid_shape: tuple[int, ...], ghost_cells: int
    ) -> tuple[int, ...]:
        """Calculate grid shape without ghost cells.

        Args:
            grid_shape: Grid shape including ghost cells.
            ghost_cells: Number of ghost cells on each side.

        Returns:
            Grid shape without ghost cells.
        """
        return tuple(
            axis - ghost_cells * 2 if axis > 1 else axis for axis in grid_shape
        )

    @staticmethod
    def get_ghost_shape(
        grid_shape: tuple[int, ...], ghost_cells: int
    ) -> tuple[int, ...]:
        """Calculate grid shape including ghost cells.

        Args:
            grid_shape: Grid shape without ghost cells.
            ghost_cells: Number of ghost cells on each side.

        Returns:
            Grid shape including ghost cells.
        """
        return tuple(
            axis + ghost_cells * 2 if axis > 1 else axis for axis in grid_shape
        )

    @setup(order=-1)
    def initialize_array(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the output array."""
        self.output_array = self.create_output_array(input_array, self.output_dtype)

    @setup(order=1)
    def initialize_kernel(self, input_array: wp.array, *args, **kwargs) -> None:
        """Initialize the kernel. Override in subclasses."""
        ...

    def forward(self, input_array: wp.array, *args, **kwargs) -> wp_Array:
        """Compute the stencil output. Must be implemented by subclasses."""
        raise NotImplementedError()

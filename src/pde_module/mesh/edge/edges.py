import numpy as np
from ..cell import Cells
from .functions import get_edges


class Edges:
    """Container for edge information.

    Edges are stored as an (E, 2) array of node IDs.

    Attributes:
        connectivity: Array of edge node pairs (E, 2).
        int_dtype: NumPy integer dtype.
        float_dtype: NumPy float dtype.
    """

    connectivity: np.ndarray
    lengths: np.ndarray
    float_dtype: np.dtype
    int_dtype: np.dtype

    def __init__(
        self, connectivity: np.ndarray, float_dtype=np.float32, int_dtype=np.int32
    ) -> None:
        """Initialize Edges object.

        Args:
            connectivity: Array of edge node pairs (E, 2).
            float_dtype: NumPy float dtype.
            int_dtype: NumPy integer dtype.
        """
        self.connectivity = connectivity
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    @classmethod
    def from_cells(cls, cells: Cells) -> "Edges":
        """Create Edges from a Cells object.

        Args:
            cells: The parent Cells object.

        Returns:
            New Edges object.
        """
        assert isinstance(cells, Cells)
        edges = get_edges(cells)
        return cls(edges, cells.float_dtype, cells.int_dtype)

    def __len__(self) -> int:
        """Return the number of edges."""
        return len(self.connectivity)

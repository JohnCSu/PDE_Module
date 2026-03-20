import numpy as np
import numba as nb
from ..cell import Cells
from .functions import get_faces


class Faces:
    """Represents faces (polygons) with VTK-style connectivity.

    Attributes:
        connectivity: Flattened face connectivity array.
        IDs: Starting offset index for each face.
        int_dtype: NumPy integer dtype.
        float_dtype: NumPy float dtype.
    """

    connectivity: np.ndarray
    IDs: np.ndarray
    float_dtype: np.dtype
    int_dtype: np.dtype

    def __init__(
        self,
        connectivity: np.ndarray,
        IDs: np.ndarray,
        float_dtype=np.float32,
        int_dtype=np.int32,
    ) -> None:
        """Initialize Faces object.

        Args:
            connectivity: Flattened face connectivity array.
            IDs: Face offset indices.
            float_dtype: NumPy float dtype.
            int_dtype: NumPy integer dtype.
        """
        self.connectivity, self.IDs = connectivity, IDs
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    @classmethod
    def from_cells(cls, cells: Cells) -> "Faces":
        """Create Faces from a Cells object.

        Args:
            cells: The parent Cells object.

        Returns:
            New Faces object.
        """
        assert isinstance(cells, Cells)
        face_connectivity, face_IDs = get_faces(cells)
        return cls(face_connectivity, face_IDs, cells.float_dtype, cells.int_dtype)

    def __len__(self) -> int:
        """Return the number of faces."""
        return len(self.IDs)


@nb.njit(cache=True)
def _calculate_faces_area_normals_centroids(
    nodes: np.ndarray, faces: np.ndarray, face_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate area, normal, and centroid for each face.

    Triangulates each face from its centroid and accumulates
    area-weighted normals.

    Args:
        nodes: Node coordinates (N, 3).
        faces: Flattened face connectivity.
        face_offsets: Starting offset for each face.

    Returns:
        Tuple of (areas, normals, centroids), each as an array.
    """
    n_faces = face_offsets.shape[0]
    areas = np.zeros(n_faces, dtype=np.float32)
    normals = np.zeros((n_faces, 3), dtype=np.float32)
    centroids = np.zeros((n_faces, 3), dtype=np.float32)

    for i in range(n_faces):
        start_idx = face_offsets[i]
        num_nodes = faces[start_idx]

        if num_nodes < 3:
            continue

        cx, cy, cz = 0.0, 0.0, 0.0
        for j in range(num_nodes):
            idx = faces[start_idx + 1 + j]
            cx += nodes[idx, 0]
            cy += nodes[idx, 1]
            cz += nodes[idx, 2]
        cx /= num_nodes
        cy /= num_nodes
        cz /= num_nodes

        centroids[i, 0] = cx
        centroids[i, 1] = cy
        centroids[i, 2] = cz

        total_area = 0.0
        nx_sum, ny_sum, nz_sum = 0.0, 0.0, 0.0

        for j in range(num_nodes):
            idx1 = faces[start_idx + 1 + j]
            idx2 = faces[start_idx + 1 + ((j + 1) % num_nodes)]

            p1x = nodes[idx1, 0] - cx
            p1y = nodes[idx1, 1] - cy
            p1z = nodes[idx1, 2] - cz

            p2x = nodes[idx2, 0] - cx
            p2y = nodes[idx2, 1] - cy
            p2z = nodes[idx2, 2] - cz

            cx_val = p1y * p2z - p1z * p2y
            cy_val = p1z * p2x - p1x * p2z
            cz_val = p1x * p2y - p1y * p2x

            mag_sq = cx_val * cx_val + cy_val * cy_val + cz_val * cz_val
            if mag_sq > 1e-12:
                mag = np.sqrt(mag_sq)
                tri_area = 0.5 * mag
                total_area += tri_area

                nx_sum += cx_val / mag * tri_area
                ny_sum += cy_val / mag * tri_area
                nz_sum += cz_val / mag * tri_area

        areas[i] = total_area

        len_sq = nx_sum * nx_sum + ny_sum * ny_sum + nz_sum * nz_sum
        if len_sq > 1e-12:
            length = np.sqrt(len_sq)
            normals[i, 0] = nx_sum / length
            normals[i, 1] = ny_sum / length
            normals[i, 2] = nz_sum / length

    return areas, normals, centroids

import warp as wp
from warp.types import vector


def create_Divergence_kernel(float_dtype):
    dimension = 3
    vec3 = vector(dimension, float_dtype)

    @wp.kernel
    def internal_divergence_kernel(
        vector_field: wp.array2d[float_dtype],
        alpha: float,
        cell_neighbors: wp.array1d[wp.vec2i],
        cell_neighbors_offsets: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        face_centroids: wp.array1d[vec3],
        cell_centroids: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        scalar_field: wp.array2d[float_dtype],
    ):
        cell_id = wp.tid()
        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        cell_volume = cell_volumes[cell_id]

        div = float_dtype(0.0)
        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[1]

            face_normal = face_normals[face_id] * wp.where(
                cell_id < neighbor_id, float_dtype(1.0), float_dtype(-1.0)
            )

            for axis in range(dimension):
                div += (
                    (vector_field[axis, cell_id] + vector_field[axis, neighbor_id])
                    / 2.0
                ) * face_normal[axis]

        scalar_field[0, cell_id] = div * alpha / cell_volume

    @wp.kernel
    def external_divergence_kernel(
        boundary_value: wp.array2d[float_dtype],
        alpha: float_dtype,
        boundary_face_to_cell_id: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        face_centroids: wp.array1d[vec3],
        cell_centroids: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        scalar_field: wp.array2d[float_dtype],
    ):
        tid = wp.tid()

        cell_id = boundary_face_to_cell_id[tid]
        face_normal = face_normals[tid]
        cell_volume = cell_volumes[cell_id]

        div = float_dtype(0.0)
        for axis in range(dimension):
            div += boundary_value[axis, tid] * face_normal[axis]
        scalar_field[0, cell_id] += div * alpha / cell_volume

    return internal_divergence_kernel, external_divergence_kernel

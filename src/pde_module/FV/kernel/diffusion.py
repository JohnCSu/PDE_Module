import warp as wp
from warp.types import vector


def create_diffusion_kernel(num_vars, float_dtype):
    vec_type = vector(num_vars, dtype=float_dtype)
    vector_type = vector(3, dtype=float_dtype)

    @wp.kernel
    def Internal_Face_Diffusion_kernel(
        cell_values: wp.array2d[float_dtype],
        cell_neighbors: wp.array1d[wp.vec2i],
        cell_neighbors_offsets: wp.array1d[int],
        face_normals: wp.array[vector_type],
        cell_centroids: wp.array[vector_type],
        cell_volumes: wp.array[float_dtype],
        viscosity: float_dtype,
        output_values: wp.array2d[float_dtype],
    ):
        cell_id = wp.tid()
        volume = cell_volumes[cell_id]

        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]

        dif = vec_type()
        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[-1]
            area = wp.length(face_normals[face_id])
            dist = wp.length(cell_centroids[neighbor_id] - cell_centroids[cell_id])
            for j in range(num_vars):
                val = (
                    (cell_values[j, neighbor_id] - cell_values[j, cell_id])
                    * area
                    / dist
                )
                dif[j] += val
        dif *= viscosity / volume
        for j in range(num_vars):
            output_values[j, cell_id] = dif[j]

    @wp.kernel
    def Boundary_Face_Diffusion_kernel(
        cell_values: wp.array2d[float_dtype],
        boundary_face_to_cell_id: wp.array1d[int],
        boundary_value: wp.array2d[float_dtype],
        face_centroids: wp.array[vector_type],
        face_normals: wp.array[vector_type],
        cell_centroids: wp.array[vector_type],
        cell_volumes: wp.array[float_dtype],
        viscosity: float_dtype,
        output_values: wp.array2d[float_dtype],
    ):
        tid = wp.tid()

        cell_id = boundary_face_to_cell_id[tid]
        volume = cell_volumes[cell_id]
        dist = wp.length(face_centroids[tid] - cell_centroids[cell_id])
        area = wp.length(face_normals[tid])
        alpha = viscosity * area / (volume * dist)

        for j in range(num_vars):
            output_values[j, cell_id] += alpha * (
                boundary_value[j, tid] - cell_values[j, cell_id]
            )

    return Internal_Face_Diffusion_kernel, Boundary_Face_Diffusion_kernel

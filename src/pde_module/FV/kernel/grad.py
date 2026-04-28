import warp as wp
from warp.types import vector


def create_grad_kernel(float_dtype):
    vec3 = vector(3, float_dtype)

    @wp.kernel
    def grad_internal_kernel(
        cell_field: wp.array2d[float_dtype],
        alpha: float,
        cell_neighbors: wp.array1d[wp.vec2i],
        cell_neighbors_offsets: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        grad_field: wp.array2d[float_dtype],
    ):
        cell_id = wp.tid()

        cell_volume = cell_volumes[cell_id]

        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        cell_volume = cell_volumes[cell_id]

        grad = vec3()

        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[1]
            face_normal = (
                wp.where(cell_id < neighbor_id, float_dtype(1.0), float_dtype(-1.0))
                * face_normals[face_id]
            )
            face_val = (
                face_normal
                * (cell_field[0, cell_id] + cell_field[0, neighbor_id])
                / 2.0
            )
            grad += face_val

        grad /= alpha * cell_volume
        for j in range(3):
            grad_field[j, cell_id] = grad[j]

    @wp.kernel
    def grad_external_kernel(
        boundary_value: wp.array2d[float_dtype],
        alpha: float_dtype,
        boundary_face_to_cell_id: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        grad_field: wp.array2d[float_dtype],
    ):
        tid = wp.tid()

        cell_id = boundary_face_to_cell_id[tid]
        face_normal = face_normals[tid]
        cell_volume = cell_volumes[cell_id]

        grad = boundary_value[0, tid] * face_normal * alpha / cell_volume
        for j in range(3):
            grad_field[j, cell_id] += grad[j]

    return grad_internal_kernel, grad_external_kernel

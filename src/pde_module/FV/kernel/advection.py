import warp as wp
from warp.types import vector


def create_advection_kernel(num_scalars, float_dtype):
    scalar_vector = vector(num_scalars, float_dtype)
    vec3 = vector(3, float_dtype)

    @wp.func
    def upwind(
        owner_value: float_dtype, neighbor_value: float_dtype, mass_flow: float_dtype
    ):
        return wp.where(mass_flow > 0.0, owner_value, neighbor_value)

    @wp.kernel
    def advection_internal_face_kernel(
        cell_field: wp.array2d[float_dtype],
        velocity_field: wp.array2d[float_dtype],
        density: float_dtype,
        cell_neighbors: wp.array1d[wp.vec2i],
        cell_neighbors_offsets: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        output_values: wp.array2d[float_dtype],
    ):
        cell_id = wp.tid()
        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        cell_volume = cell_volumes[cell_id]
        advec = scalar_vector()
        u_vec = vec3()
        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[1]

            for jj in range(3):
                u_vec[jj] = (
                    velocity_field[jj, cell_id] + velocity_field[jj, neighbor_id]
                ) / 2.0

            face_normal = (
                wp.where(cell_id < neighbor_id, 1.0, -1.0) * face_normals[face_id]
            )
            mass_flow = density * wp.dot(face_normal, u_vec)

            for scalar in range(num_scalars):
                advec[scalar] = (
                    advec[scalar]
                    + upwind(
                        cell_field[scalar, cell_id],
                        cell_field[scalar, neighbor_id],
                        mass_flow,
                    )
                    * mass_flow
                )

        advec /= cell_volume
        for scalar in range(num_scalars):
            output_values[scalar, cell_id] = advec[scalar]

    @wp.kernel
    def advection_external_face_kernel(
        boundary_value: wp.array2d[float_dtype],
        velocity_boundary: wp.array2d[float_dtype],
        density: float_dtype,
        boundary_face_to_cell_id: wp.array1d[int],
        face_normals: wp.array1d[vec3],
        cell_volumes: wp.array1d[float_dtype],
        output_values: wp.array2d[float_dtype],
    ):
        tid = wp.tid()
        cell_id = boundary_face_to_cell_id[tid]
        u_vec = vec3()
        cell_volume = cell_volumes[cell_id]
        for jj in range(3):
            u_vec[jj] = velocity_boundary[jj, tid]
        mass_flow = density * wp.dot(face_normals[tid], u_vec) / cell_volume

        for scalar in range(num_scalars):
            output_values[scalar, cell_id] = (
                output_values[scalar, cell_id] + mass_flow * boundary_value[scalar, tid]
            )

    return advection_internal_face_kernel, advection_external_face_kernel

import warp as wp
from warp.types import vector
from pde_module.FV.flags import DIRICHLET, VON_NEUMANN


def create_FV_Boundary_kernel(num_vars, float_dtype):
    vec3 = vector(3, float_dtype)

    @wp.kernel
    def boundary_kernel(
        cell_field: wp.array2d[float_dtype],
        face_to_cell_id: wp.array1d[int],
        boundary_values: wp.array2d[float_dtype],
        boundary_types: wp.array2d[wp.uint8],
        exterior_face_centroid: wp.array1d[vec3],
        cell_centroids: wp.array1d[vec3],
        boundary_face_field: wp.array2d[float_dtype],
    ):
        tid = wp.tid()

        cell_id = face_to_cell_id[tid]

        for var in range(num_vars):
            BC_type = boundary_types[var, tid]
            BC_value = boundary_values[var, tid]
            if BC_type == DIRICHLET:
                boundary_face_field[var, tid] = BC_value
            else:
                boundary_face_field[var, tid] = cell_field[
                    var, cell_id
                ] + BC_value / wp.length(
                    exterior_face_centroid[tid] - cell_centroids[tid]
                )

    return boundary_kernel

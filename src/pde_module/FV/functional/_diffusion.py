import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def diffusion(
    internal_kernel: wp_Kernel,
    external_kernel: wp_Kernel,
    input_field: wp_Array,
    boundary_values: wp_Array,
    viscosity: float,
    neighbors: wp_Array,
    neighbors_offset: wp_Array,
    face_normals: wp_Array,
    cell_centroids: wp_Array,
    cell_volumes: wp_Array,
    exterior_cell_ids: wp_Array,
    exterior_centroids: wp_Array,
    exterior_normals: wp_Array,
    output_field: wp_Array,
    num_internal_faces: int,
    num_boundary_faces: int,
    device=None,
) -> wp_Array:
    wp.launch(
        internal_kernel,
        dim=num_internal_faces,
        inputs=[
            input_field,
            neighbors,
            neighbors_offset,
            face_normals,
            cell_centroids,
            cell_volumes,
            viscosity,
        ],
        outputs=[output_field],
        device=device,
    )
    wp.launch(
        external_kernel,
        dim=num_boundary_faces,
        inputs=[
            input_field,
            exterior_cell_ids,
            boundary_values,
            exterior_centroids,
            exterior_normals,
            cell_centroids,
            cell_volumes,
            viscosity,
        ],
        outputs=[output_field],
        device=device,
    )
    return output_field

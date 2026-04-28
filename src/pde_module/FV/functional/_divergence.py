import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def divergence(
    internal_kernel: wp_Kernel,
    external_kernel: wp_Kernel,
    vector_field: wp_Array,
    boundary_values: wp_Array,
    alpha: float,
    neighbors: wp_Array,
    neighbors_offset: wp_Array,
    face_normals: wp_Array,
    face_centroids: wp_Array,
    cell_centroids: wp_Array,
    cell_volumes: wp_Array,
    exterior_cell_ids: wp_Array,
    exterior_normals: wp_Array,
    output_field: wp_Array,
    num_cells: int,
    num_boundary_faces: int,
    device=None,
) -> wp_Array:
    wp.launch(
        internal_kernel,
        dim=num_cells,
        inputs=[
            vector_field,
            alpha,
            neighbors,
            neighbors_offset,
            face_normals,
            face_centroids,
            cell_centroids,
            cell_volumes,
        ],
        outputs=[output_field],
        device=device,
    )
    wp.launch(
        external_kernel,
        dim=num_boundary_faces,
        inputs=[
            boundary_values,
            alpha,
            exterior_cell_ids,
            exterior_normals,
            face_centroids,
            cell_centroids,
            cell_volumes,
        ],
        outputs=[output_field],
        device=device,
    )
    return output_field

import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def boundary(
    kernel: wp_Kernel,
    cell_field: wp_Array,
    cell_ids: wp_Array,
    boundary_values: wp_Array,
    boundary_types: wp_Array,
    exterior_centroids: wp_Array,
    cell_centroids: wp_Array,
    boundary_field: wp_Array,
    num_faces: int,
    device=None,
) -> wp_Array:
    wp.launch(
        kernel,
        dim=num_faces,
        inputs=[
            cell_field,
            cell_ids,
            boundary_values,
            boundary_types,
            exterior_centroids,
            cell_centroids,
        ],
        outputs=[boundary_field],
        device=device,
    )
    return boundary_field

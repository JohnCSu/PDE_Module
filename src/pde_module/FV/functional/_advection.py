import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def advection(
    internal_kernel: wp_Kernel,
    external_kernel: wp_Kernel,
    scalar_cell_field: wp_Array,
    scalar_boundary_values: wp_Array,
    velocity_cell_field: wp_Array,
    velocity_boundary_values: wp_Array,
    density: float,
    neighbors: wp_Array,
    neighbors_offset: wp_Array,
    face_normals: wp_Array,
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
            scalar_cell_field,
            velocity_cell_field,
            density,
            neighbors,
            neighbors_offset,
            face_normals,
            cell_volumes,
        ],
        outputs=[output_field],
        device=device,
    )
    wp.launch(
        external_kernel,
        dim=num_boundary_faces,
        inputs=[
            scalar_boundary_values,
            velocity_boundary_values,
            density,
            exterior_cell_ids,
            exterior_normals,
            cell_volumes,
        ],
        outputs=[output_field],
        device=device,
    )
    return output_field

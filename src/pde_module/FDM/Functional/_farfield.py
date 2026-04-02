import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def farfield(
    kernel: wp_Kernel,
    input_array: wp_Array,
    warp_outer_points: wp_Array,
    farfield_condition,
    sigma_max: float,
    output_array: wp_Array,
    device=None,
) -> wp_Array:
    """Apply the far-field sponge layer boundary condition.

    Args:
        kernel: The sponge layer kernel to use.
        input_array: 3D array matching input shape.
        warp_outer_points: Array of grid point indices for the sponge layer.
        farfield_condition: Value that the solution is forced toward.
        sigma_max: Scaling factor.
        output_array: Pre-allocated output array.
        device: Optional device for kernel launch.

    Returns:
        3D array with same shape and dtype as input_array.
    """
    wp.launch(
        kernel,
        dim=len(warp_outer_points),
        inputs=[
            input_array,
            warp_outer_points,
            farfield_condition,
            sigma_max,
        ],
        outputs=[output_array],
        device=device,
    )
    return output_array

import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def viscousDampingLayer(
    kernel: wp_Kernel,
    du_dt: wp_Array,
    warp_outer_points: wp_Array,
    C_max: float,
    output_array: wp_Array,
    device=None,
) -> wp_Array:
    """Apply the viscous damping layer.

    Args:
        kernel: The damping layer kernel to use.
        du_dt: 3D array representing the time derivative.
        warp_outer_points: Array of grid point indices for the damping layer.
        C_max: Scaling factor for the damping.
        output_array: Pre-allocated output array.
        device: Optional device for kernel launch.

    Returns:
        3D array with same shape and dtype as input.
    """
    wp.launch(
        kernel,
        dim=len(warp_outer_points),
        inputs=[du_dt, warp_outer_points, C_max],
        outputs=[output_array],
        device=device,
    )
    return output_array

import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def streaming(
    kernel: wp_Kernel,
    f_in: wp_Array,
    f_out: wp_Array,
    grid_shape: tuple[int],
    device=None,
) -> wp_Array:
    """Apply streaming operation.

    Args:
        kernel: The streaming kernel.
        f_in: Input distribution functions array.
        f_out: Pre-allocated output array.
        grid_shape: Shape of the grid.
        device: Optional device for kernel launch.

    Returns:
        Output distribution functions array after streaming.
    """
    wp.launch(
        kernel=kernel, dim=grid_shape, inputs=[f_in], outputs=[f_out], device=device
    )
    return f_out

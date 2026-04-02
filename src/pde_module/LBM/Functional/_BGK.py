import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def bgk_collision(
    kernel: wp_Kernel,
    f_in: wp_Array,
    tau: float,
    f_out: wp_Array,
    device=None,
) -> wp_Array:
    """Apply BGK collision operator.

    Args:
        kernel: The BGK collision kernel.
        f_in: Input distribution functions array.
        tau: Relaxation time.
        f_out: Pre-allocated output array.
        device: Optional device for kernel launch.

    Returns:
        Output distribution functions array after collision.
    """
    wp.launch(
        kernel,
        dim=f_in.shape[-1],
        inputs=[f_in, 1.0 / tau],
        outputs=[f_out],
        device=device,
    )
    return f_out

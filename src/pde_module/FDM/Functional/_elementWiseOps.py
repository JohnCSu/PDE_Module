import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def elementWiseOp(
    kernel: wp_Kernel,
    array_A: wp_Array,
    array_B: wp_Array,
    output_array: wp_Array,
    device=None,
) -> wp_Array:
    """Apply element-wise operation to arrays A and B.

    Args:
        kernel: The element-wise operation kernel.
        array_A: First input array.
        array_B: Second input array with same shape as array_A.
        output_array: Pre-allocated output array.
        device: Optional device for kernel launch.

    Returns:
        Output array containing the element-wise result.
    """
    dim = array_A.size
    wp.launch(
        kernel,
        dim=dim,
        inputs=[array_A.flatten(), array_B.flatten()],
        outputs=[output_array.flatten()],
        device=device,
    )
    return output_array

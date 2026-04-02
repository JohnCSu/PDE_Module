import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def laplacian(
    kernel: wp_Kernel,
    input_array: wp_Array,
    alpha: float,
    output_array: wp_Array,
    ghost_cells: int,
    device=None,
) -> wp_Array:
    """Compute the Laplacian of the input field.

    Args:
        kernel: The Laplacian kernel to use.
        input_array: 3D array with vector dtype representing the field.
        alpha: Scaling factor for the Laplacian term.
        output_array: Pre-allocated output array.
        ghost_cells: Number of ghost cells.
        device: Optional device for kernel launch.

    Returns:
        3D array with same vector dtype representing the Laplacian.
    """
    dims = tuple(
        axis - ghost_cells * 2 if axis > 1 else axis for axis in input_array.shape
    )
    wp.launch(
        kernel,
        dim=dims,
        inputs=[input_array, alpha],
        outputs=[output_array],
        device=device,
    )
    return output_array

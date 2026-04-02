import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def boundary(
    kernel: wp_Kernel,
    f_in: wp_Array,
    warp_flags: wp_Array,
    warp_indices: wp_Array,
    warp_BC_velocity: wp_Array,
    warp_BC_density: wp_Array,
    f_out: wp_Array,
    device=None,
) -> wp_Array:
    """Apply boundary conditions.

    Args:
        kernel: The boundary condition kernel.
        f_in: Input distribution functions array.
        warp_flags: Boundary flags array.
        warp_indices: Indices of boundary nodes.
        warp_BC_velocity: Boundary condition velocities.
        warp_BC_density: Boundary condition densities.
        f_out: Pre-allocated output array.
        device: Optional device for kernel launch.

    Returns:
        Output distribution functions array after applying BCs.
    """
    wp.copy(f_out, f_in)
    wp.launch(
        kernel,
        len(warp_indices),
        [f_in, warp_flags, warp_indices, warp_BC_velocity, warp_BC_density],
        outputs=[f_out],
        device=device,
    )
    return f_out

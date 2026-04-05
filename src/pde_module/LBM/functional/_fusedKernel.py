import warp as wp
from pde_module.utils.types import wp_Array, wp_Kernel


def fused_lbm_kernel(
    kernel: wp_Kernel,
    f_in: wp_Array,
    tau: float,
    warp_flags: wp_Array,
    warp_BC_velocity: wp_Array,
    warp_BC_density: wp_Array,
    sigma:float,
    ramp:float,
    f_out: wp_Array,
    num_nodes: int,
    device=None,
) -> wp_Array:
    """Apply fused LBM kernel (streaming + collision + boundary).

    Args:
        kernel: The fused LBM kernel.
        f_in: Input distribution functions array.
        tau: Relaxation time.
        warp_flags: Boundary flags array.
        warp_BC_velocity: Boundary condition velocities.
        warp_BC_density: Boundary condition densities.
        f_out: Pre-allocated output array.
        num_nodes: Number of nodes to process.
        device: Optional device for kernel launch.

    Returns:
        Output distribution functions array.
    """
    wp.launch(
        kernel,
        dim=num_nodes,
        inputs=[
            f_in,
            1.0 / tau,
            warp_flags,
            warp_BC_velocity,
            warp_BC_density,
            sigma,
            ramp,
        ],
        outputs=[f_out],
        device=device,
    )
    return f_out

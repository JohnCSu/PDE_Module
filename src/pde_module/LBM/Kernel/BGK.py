import warp as wp
from warp.types import vector
from pde_module.utils.types import wp_Dtype
from typing import Callable


def create_BGK_feq(dimension: int, float_dtype: wp_Dtype) -> Callable:
    @wp.func
    def BGK_feq(
        weight: float_dtype,
        rho: float_dtype,
        u: vector(dimension, float_dtype),
        ei: vector(dimension, float_dtype),
    ):
        ei_dot_u = wp.dot(ei, u)
        return (
            weight
            * rho
            * (1.0 + 3.0 * ei_dot_u + 4.5 * ei_dot_u * ei_dot_u - 1.5 * wp.dot(u, u))
        )

    return BGK_feq


def create_BGK_collision(
    float_velocity_directions,
    weights,
    num_distributions,
    dimension,
    grid_shape,
    float_dtype,
):
    grid_shape = wp.vec3i(grid_shape)
    BGK_feq = create_BGK_feq(dimension, float_dtype)

    @wp.kernel
    def BGK_collision_kernel(
        f_in: wp.array2d(dtype=float_dtype),
        inv_tau: float_dtype,
        f_out: wp.array2d(dtype=float_dtype),
    ):
        global_id = wp.tid()

        rho = float_dtype(0.0)
        u = vector(float_dtype(0.0), length=dimension)
        for f in range(num_distributions):
            rho += f_in[f, global_id]
            u += f_in[f, global_id] * float_velocity_directions[f]
        u /= rho

        for f in range(num_distributions):
            feq = BGK_feq(weights[f], rho, u, float_velocity_directions[f])
            f_out[f, global_id] = f_in[f, global_id] - inv_tau * (
                f_in[f, global_id] - feq
            )

    return BGK_collision_kernel

from typing import NewType, Any,TypeVar
import numpy as np
from warp import Kernel,Function

wp_Vector = NewType("wp_Vector", np.ndarray)
"""Dummy type for warp vectors."""

wp_Matrix = NewType("wp_Matrix", np.ndarray)
"""Dummy type for warp matrices."""

wp_Array = NewType("wp_Array", np.ndarray)
"""Dummy type for warp arrays."""

wp_Vec3i = NewType("wp_Vec3i", np.ndarray)
"""Dummy type for warp vec3i."""

wp_Vec2i = NewType("wp_Vec2i", np.ndarray)
"""Dummy type for warp vec2i."""

wp_Kernel = NewType("wp_Kernel", Kernel)
"""Dummy type for warp kernels."""

wp_Function = NewType("wp_Function", Function)
"""Dummy type for warp functions."""

__all__ = [
    "wp_Vector",
    "wp_Matrix",
    "wp_Array",
    "wp_Vec3i",
    "wp_Vec2i",
    "wp_Kernel",
    "wp_Function",
    "Any",
]

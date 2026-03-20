from typing import NewType, Any


wp_Vector = NewType("wp_Vector", int)
"""Dummy type for warp vectors."""

wp_Matrix = NewType("wp_Matrix", int)
"""Dummy type for warp matrices."""

wp_Array = NewType("wp_Array", int)
"""Dummy type for warp arrays."""

wp_Vec3i = NewType("wp_Vec3i", int)
"""Dummy type for warp vec3i."""

wp_Vec2i = NewType("wp_Vec2i", int)
"""Dummy type for warp vec2i."""

wp_Kernel = NewType("wp_Kernel", int)
"""Dummy type for warp kernels."""

wp_Function = NewType("wp_Function", int)
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

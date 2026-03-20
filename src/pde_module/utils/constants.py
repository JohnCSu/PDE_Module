import warp as wp
from enum import IntEnum

INT32_MAX = wp.int32(2**31 - 1)
"""Maximum value for 32-bit signed integer."""

INT32_MIN = wp.int32(-(2**31))
"""Minimum value for 32-bit signed integer."""

NO_BC = wp.int8(0)
"""No boundary condition flag."""

DIRICHLET = wp.int8(1)
"""Dirichlet boundary condition flag."""

VON_NEUMANN = wp.int8(2)
"""Von Neumann boundary condition flag."""


class Boundary_Types(IntEnum):
    """Enumeration of boundary condition types."""

    NO_BC = NO_BC
    DIRICHLET = DIRICHLET
    VON_NEUMANN = VON_NEUMANN


FLUID_CELL = wp.int8(0)
"""Flag indicating a fluid cell in the domain."""

SOLID_CELL = wp.int8(1)
"""Flag indicating a solid cell in the domain."""

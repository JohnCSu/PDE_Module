import warp as wp
INT32_MAX = wp.int32(2**31 - 1)
"""Maximum value for 32-bit signed integer."""

INT32_MIN = wp.int32(-(2**31))
"""Minimum value for 32-bit signed integer."""

NO_BC = 0
"""No boundary condition flag."""

DIRICHLET = 1
"""Dirichlet boundary condition flag."""

VON_NEUMANN = 2
"""Von Neumann boundary condition flag."""




FLUID_CELL = 0
"""Flag indicating a fluid cell in the domain."""

SOLID_CELL = 1
"""Flag indicating a solid cell in the domain."""

import warp as wp
from enum import Enum,IntEnum
#Int Maxs and mins
INT32_MAX = wp.int32(2**31 -1)
INT32_MIN = wp.int32(-2**31)

# BC
NO_BC = wp.int8(0)
DIRICHLET = wp.int8(1)
VON_NEUMANN = wp.int8(2)


class Boundary_Types(IntEnum):
    NO_BC = NO_BC
    DIRICHLET = DIRICHLET
    VON_NEUMANN = VON_NEUMANN

# Cell State
FLUID_CELL = wp.int8(0)
SOLID_CELL = wp.int8(1)
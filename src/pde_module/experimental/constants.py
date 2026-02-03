import warp as wp

#Int Maxs and mins
INT32_MAX = wp.int32(2**31 -1)
INT32_MIN = wp.int32(-2**31)

# BC
DIRICHLET = wp.int8(1)
VON_NEUMANN = wp.int8(2)


# Cell State
FLUID_CELL = wp.int8(0)
SOLID_CELL = wp.int8(1)
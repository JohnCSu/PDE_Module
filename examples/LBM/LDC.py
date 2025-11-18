from pde_module.stencils.LBM.kernels import create_streaming_kernel,create_BGK_kernel
from pde_module.stencils.LBM.modules.gridBoundary import LBMGridBoundary
import warp as wp
from pde_module.grids import Grid
from pde_module.grids.latticeModels import D2Q9_Model 
import numpy as np
wp.config.mode = 'debug'
wp.init()


n = 101 # Use Odd number of points!
dx = 1/(n-1)
LBM = Grid('cell',dx=dx,nx = n,ny = n,levels = 1)
LBM.set_LBM(D2Q9_Model(),density= 1, dynamic_viscosity= 0.01,u_ref = 1.,u_target= 0.1)
t = 0

dx = LBM.dx
Re = 100.
viscosity = 1/Re
beta = 0.7
CFL_LIMIT = 0.5

dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
print(f'{dt=:.3E} {beta=:.3E} {viscosity=:.3E}')


f_0 = LBM.create_cell_field(9)
f_1 = LBM.create_cell_field(9)
f_2 = LBM.create_cell_field(9)

vel_field = LBM.create_cell_field(2)
rho_field = LBM.create_cell_field(1)

D2Q9 = LBM.LBM_lattice

boundary = LBMGridBoundary(LBM,dynamic_array_alloc=False)


boundary.no_slip('ALL')
boundary.moving_wall('+Y',1.,axis = 0)

BGK_kernel = create_BGK_kernel(LBM.dimension,D2Q9.num_discrete_velocities)
stream_kernel = create_streaming_kernel(LBM.dimension,D2Q9.num_discrete_velocities)


wp.launch(BGK_kernel,dim = f_0.shape ,inputs=[f_0,rho_field,vel_field,D2Q9.velocity_directions_float,D2Q9.weights,0.5],outputs=[f_1])
wp.launch(stream_kernel,dim = f_1.shape,inputs=[f_1,D2Q9.velocity_directions_int],outputs=[f_2])


# a = GridBoundary(LBM)
# a.define_indices()

# print(a.boundaries['-X'].shape)
# print(a.boundaries['-Y'].shape)
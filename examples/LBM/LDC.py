from pde_module.stencils.LBM.kernels import create_streaming_kernel,create_BGK_kernel
from pde_module.stencils.LBM.modules.gridBoundary import GridBoundary
import warp as wp
from pde_module.grids.cell_grid import UniformCellGrid

wp.config.mode = 'debug'
wp.init()

LBM = UniformCellGrid(0.1,2,2)
LBM.LBM()


f_old = LBM.generate_distribution_field()
f_new = LBM.generate_distribution_field()
vel_field = LBM.generate_velocity_field()
rho_field = LBM.generate_density_field()

BGK_kernel = create_BGK_kernel(LBM.dimension,LBM.num_discrete_velocities)
stream_kernel = create_streaming_kernel(LBM.dimension,LBM.num_discrete_velocities)


wp.launch(BGK_kernel,dim = LBM.batch_shape(1),inputs=[f_old,rho_field,vel_field,LBM.velocity_directions_float,LBM.weights,0.5],outputs=[f_new])
wp.launch(stream_kernel,dim = LBM.batch_shape(1),inputs=[f_old,LBM.velocity_directions_int],outputs=[f_new])


a = GridBoundary(LBM)
a.define_indices()

print(a.boundaries['-X'].shape)
print(a.boundaries['-Y'].shape)
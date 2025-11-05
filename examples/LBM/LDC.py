from pde_module.stencils.LBM.kernels import create_streaming_kernel,create_BGK_kernel
from pde_module.stencils.LBM.modules.gridBoundary import GridBoundary
import warp as wp
from pde_module.grids import Grid
import numpy as np
wp.config.mode = 'debug'
wp.init()


class D2Q9_Model:
    def __init__(self):
        self.num_discrete_velocities = 9
        
        self.D2Q9_velocities = np.array([
            [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
            [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
        ]).swapaxes(0,1)
        self.velocity_directions_int =wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = int)(self.D2Q9_velocities)
        
        self.velocity_directions_float = wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = float)(self.D2Q9_velocities.astype( self.float_dtype))
        
        
        self.indices = wp.vec(length = self.num_discrete_velocities,dtype = int)(np.arange(self.num_discrete_velocities,dtype= int))
        
        self.weights = wp.vec(length = self.num_discrete_velocities,dtype = wp.dtype_from_numpy(self.float_dtype) )([
                                4/9,                        # Center Velocity [0,]
                                1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
                                1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
                                ])


n = 101 # Use Odd number of points!
dx = 1/(n-1)
LBM = Grid('cell',dx=dx,nx = n,ny = n,levels = 1)
D2Q9 = D2Q9_Model()
t = 0

dx = LBM.dx
Re = 100.
viscosity = 1/Re
beta = 0.7
CFL_LIMIT = 0.5

dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
print(f'{dt=:.3E} {beta=:.3E} {viscosity=:.3E}')


f_old = LBM.create_cell_field(9)
f_new = LBM.create_cell_field(9)
vel_field = LBM.create_cell_field(3)
rho_field = LBM.create_cell_field(1)

BGK_kernel = create_BGK_kernel(LBM.dimension,D2Q9.num_discrete_velocities)
stream_kernel = create_streaming_kernel(LBM.dimension,D2Q9.num_discrete_velocities)


wp.launch(BGK_kernel,dim = ,inputs=[f_old,rho_field,vel_field,D2Q9.velocity_directions_float,D2Q9.weights,0.5],outputs=[f_new])
wp.launch(stream_kernel,dim = ,inputs=[f_old,D2Q9.velocity_directions_int],outputs=[f_new])


a = GridBoundary(LBM)
a.define_indices()

print(a.boundaries['-X'].shape)
print(a.boundaries['-Y'].shape)
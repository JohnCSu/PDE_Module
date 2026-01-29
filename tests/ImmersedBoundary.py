
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.experimental.grid import Grid
from pde_module.experimental.FDM.laplacian import Laplacian
from pde_module.experimental.time_integrators import ForwardEuler
from pde_module.experimental.FDM.gridBoundary import GridBoundary
from pde_module.experimental.FDM.immersedBoundary import ImmersedBoundary
from pde_module.experimental.FDM.grad import Grad
from pde_module.experimental.FDM.divergence import Divergence


wp.init()
wp.config.mode = "debug"

if __name__ == '__main__':
    n = 4
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    u = grid.create_node_field(1)
    
    R = 0.25
    func = lambda x,y,z : (x-0.5)**2 + (y-0.5)**2 <= R**2
    
    cyl = ImmersedBoundary(u,dx,ghost_cells)
    # print(cyl.conv_kernel.squeeze())
    # print(cyl.conv_kernel.shape)
    
    meshgrid = grid.create_meshgrid('node')
    cyl.from_bool_func(func,meshgrid)
    print(cyl.bitmask.squeeze())
    cyl.finalize()
    cyl.setup(u)
    cyl(u)
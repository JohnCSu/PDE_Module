import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.experimental.grid import Grid
from pde_module.experimental.FDM.dampingLayer import DampingLayer
        


if __name__ == '__main__':
    wp.init()
    wp.config.mode = "debug"

    n = 7
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    u = grid.create_node_field(2)
    u.fill_(wp.vec2f(2.,2.))
    
    
    num_layers = 3
    damp = DampingLayer(2,num_layers,2.,u.shape,dx,ghost_cells)
    
    H,W,_ = u.shape
    
    
    x1 = 2*(num_layers*W)+ (H-2*num_layers)*num_layers*2 
    x2 = len(damp.outer_layers)
    print(x1,x2)
    assert x1 == x2 == 72
    
    bitmask = np.zeros((H,W),dtype=np.bool)
    bitmask[damp.outer_layers[:,0],damp.outer_layers[:,1]] = True
    print(bitmask)
    
    np.set_printoptions(precision=3,suppress= False)
    
    out = damp(u,wp.vec2f(3.),1.)
    print(out.numpy().squeeze()[:,:,0])

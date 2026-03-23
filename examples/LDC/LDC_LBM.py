from pde_module.LBM import BGK_collision,Streaming
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.LBM import LBM_Mesh
import pyvista as pv


wp.init()
wp.config.mode = "debug"

if __name__ == '__main__':
    n = 5
    L = 1
    dx = L/(n-1)
    lbm = LBM_Mesh('D2Q9',dx,nodes_per_axis=(n,n,1),origin=(0.,0.,0.))
    # Runtime params
    viscosity = 1/100
    beta = 0.3
    CFL_LIMIT = 0.5
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    
    # Define Modules
    # f_dist = lbm.create_field(9,0.1)
    np.set_printoptions(precision=2)
    
    f_dist =  np.tile(np.arange(0,n-1,1,dtype=np.float32), (9, n-1, 1))
    f_dist = np.random.rand(9,n-1,n-1,1)
    i = 1
    print('initial')
    print(f_dist[0].squeeze())
    print(f_dist[2].squeeze())
    
    
    flatten = f_dist.reshape((9,-1))
    f_dist = wp.array(flatten)
    
   
    stream = Streaming.from_LBM_Mesh(lbm)
    collide = BGK_collision.from_LBM_Mesh(lbm)
    
    
    for i in range(1):
        f_stream = stream(f_dist) 

    
    x = f_stream.numpy().reshape((9,n-1,n-1))
    print('streamed')
    # print(f_stream.numpy()[i]
    print(x[0]) # Should be same
    print(x[2])
    
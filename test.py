import numpy as np
import warp as wp
from src.stencils.functional.neighbors import get_adjacent_points_2D_function
from src.stencils.functional import first_order,second_order
from src.stencils.functional.operators import laplacian2D
from src.stencils.functional.time_integrators import forward_euler
wp.init()
wp.config.mode = "debug"
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque
from src.grids import NodeGrid


if __name__ == '__main__':
    
    x,y = np.linspace(0,1,100),np.linspace(0,1,100)
    grid = NodeGrid(x,y,num_outputs=1)

    IC = lambda x: (np.sin(np.pi*x[:,:,:,0])*np.sin(np.pi*x[:,:,:,1]))[:,:,:,np.newaxis]
    initial_value =grid.initial_condition(IC)
   
   
    levels = np.linspace(-0.15,1.05,100,endpoint=True)
    plt.contourf(*grid.plt_meshgrid,initial_value[0,:,:,0,0],cmap = 'jet',levels = levels)
    plt.colorbar()
    plt.show()

    t = 0
    
    dx = x[1] - x[0]
    dt = float(dx**2/(4*0.1))
    grid.to_warp()
    
    
    print(f't = {t:.3e} max value = {np.max(grid.values[0].numpy().max()):.3E}, dt = {dt:.3E}')
    for i in range(1000):
        new_values = wp.clone(grid.values[0])
        laplacian = laplacian2D(grid,grid.values[0],new_values,alpha = 0.1,dimension = 2 )
        
        grid.values[1] = forward_euler(grid,grid.values[0],laplacian,dt)
        
        grid.rotate_values()
        print(f't = {t:.3e} max value = {np.max(grid.values[0].numpy().max()):.3E}, dt = {dt:.3E}')
        t += dt
        
        if t > 0.5:
            break
        
    
    
    grid.to_numpy()
    levels = np.linspace(-0.15,1.05,100,endpoint=True)
    plt.contourf(*grid.plt_meshgrid,initial_value[0,:,:,0,0],cmap = 'jet',levels = levels)
    plt.colorbar()
    plt.show()

        
        
        

    

    
    
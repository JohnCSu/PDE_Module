'''
LID DRIVEN CAVITY

This examples shows solving 2D Navier Stokes in the classic LDC example using the Artificial Compressibility Method for explicit CFD. 

No GRAPH CAPTURE IS USED HERE. Compare with examples/Benchmark/LDC_Graph_Capture.py to see how graph capture can significantly speed up runtimes.

Domain: 101 x 101 grid nodes
Re: 100 (All other constant are 1)

Equations:

du/dt grad_u * u =  d^2u/dx^2 + d^2u/dy^2 - dp/dx

dp/dt +beta * div(u) = 0


Constants to tune:
dt - time step. Simple Euler Step is used
beta =0.3  ACM term, higher means faster convergence but instability (akin to increasing time stepping)
CFL_LIMIT = 0.5 - constant to ensure CFL is not violated set to 0.5

'''

import numpy as np
import warp as wp
from matplotlib import pyplot as plt
import pyvista as pv
from pde_module.time_step.forwardEuler import ForwardEuler
from pde_module.deprecated.gridBoundary import GridBoundary
from pde_module.FDM import (Laplacian,
                            Grad,
                            GridBoundary,
                            Divergence,)
from pde_module.mesh import UniformGridMesh,create_structured_warp_field,to_pyvista

wp.init()
# wp.config.mode = "debug"

def LDC_FDM():
    n = 21
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    # grid = Grid(dx = dx,num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    grid = UniformGridMesh(dx,nodes_per_axis=(n,n,1),origin=(0.,0.,0.),ghost_cells=1)
    # Runtime params
    viscosity = 1/100
    beta = 0.3
    CFL_LIMIT = 0.5
    dt = min(CFL_LIMIT*float(dx**2/(viscosity)),CFL_LIMIT*dx/(1+1/beta))
    
    # Define Modules
    
    u = create_structured_warp_field(grid,'node',2)
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.dirichlet_BC('ALL',0.)
    u_BC.dirichlet_BC('+Y',1.,0)    
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    
    u_div = Divergence(2,u.shape,dx,ghost_cells)
    u_step = ForwardEuler(u.dtype)
    
    p = create_structured_warp_field(grid,'node',1)
    p_BC = GridBoundary(p,dx,ghost_cells)
    p_BC.vonNeumann_BC('ALL',0.)
    p_grad = Grad(1,p.shape,dx,ghost_cells)
    p_step = ForwardEuler(p.dtype)
    t= 0
    '''
    Equations:

    du/dt = d^2u/dx^2 - dp/dx
    dv/dt = d^2v/dx^2 - dp/dx

    dp/dt +beta * div(u) = 0
    '''
    # Inital Pass + Setup Of kernels
    wp.set_mempool_release_threshold("cuda:0", 0.25)
    u_fix = u_BC(u)
    u_diff = u_lapl(u_fix,viscosity)
    du = u_grad(u_fix)
    u_incomp = u_div(u_fix,-beta)
    
    u_conv = du*u
    
    p_fix = p_BC(p)
    dp = p_grad(p_fix,-1.)
    u_F = dp - u_conv + u_diff
    u_next = u_step(u_fix,u_F,dt)
    p_next = p_step(p_fix,u_incomp,dt)
    u,p = u_next,p_next
    t+= dt
    
    with wp.ScopedTimer("LDC No Graph"):
        for i in range(1,10):
            u_fix = u_BC(u)
            u_diff = u_lapl(u_fix,viscosity)
            du = u_grad(u_fix)
            u_incomp = u_div(u_fix,-beta)
            u_conv = du*u
            p_fix = p_BC(p)
            dp = p_grad(p_fix,-1.)
            u_F = dp - u_conv + u_diff
            u_next = u_step(u_fix,u_F,dt)
            p_next = p_step(p_fix,u_incomp,dt)
            u,p = u_next,p_next
            t+= dt
    return True
import pytest
# @pytest.fixture(scope="function")
def test_function():
    assert LDC_FDM()
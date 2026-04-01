'''
Transient lid driven cavity for Re = 100 With RungeKatta 2nd order time stepping. This shows hwo to implement RK2 and use matplotlib for animations

Rungekatta uses 2 evaulations per time step but is o(dt**2) accurate, making it worth while compared to the the standard forward euler step.
In this example, the CFL limit can be increased to 0.8 with no issues compared to the forward euler where the max time step is around a CFL limit
of 0.2. This is about a 3-4x increase in time stepping for the same number of evaluations 

Using weakly compressible flow, we demonstrate the LDC problem as a time dependent flow.

The 2 fields to solve are the velocity and density (rho) fields.

Here we solve the conservative momentum equations where m is the target of interest rather than solving directly u as density is not a constant:

m_{i} = rho_{i}*u_{i}
p_{i} = cs**2(rho_{i} - rho_0)

m_{i+1} = m_{i} + dt*(-dp - u_conv + u_diff - u_damp)_{i}
rho_{i+1} = rho_{i} - dt*(div(m))

u_{i+1} = m_{i+1}/rho_{i+1}

Here:
- u_conv = div(m outer u)
- u_damp is biharmonic damping applied to all fluid nodes to prevent pressure checkerboarding

Geometry Params:
Domain = 1x1
n = 201
Runtime Params
U = 1.
Re = 100
rho_0 = 1.
M = 0.1
CFL_limit = 0.1
biharmonic_eps = 0.01

'''
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.mesh import UniformGridMesh,create_structured_warp_field,to_pyvista
import pyvista as pv
from pde_module.FDM import (Laplacian,
                            Grad,
                            GridBoundary,
                            Divergence,
                            scalarVectorMult,
                            OuterProduct)

from pde_module.stencil import MapWise
from warp.types import vector

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pde_module.time_step.rungeKatta import RungeKatta2



wp.init()
# wp.config.mode = "debug"

def LBM_RK2():
    np.set_printoptions(precision=2,suppress= False)
    n = 21
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    U = 1.
    D = 1.
    R = D/2
    grid = UniformGridMesh(dx = dx,nodes_per_axis=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    # Runtime params
    viscosity = 1/100
    density = 1.
    M = 0.1
    cs = U/M
    c2 = cs**2
    
    CFL_LIMIT = 0.80
    
    d_vis = CFL_LIMIT*float(dx**2/(viscosity))
    d_conv = CFL_LIMIT*dx/cs
    d_acoustic = CFL_LIMIT*dx/(cs + U)
    
    dt = min(d_vis,d_conv,d_acoustic)
    damping_eps = -0.01*(np.exp( 4*np.log(dx) - np.log(dt))/viscosity) # Use Logs for better numerical stability
    print(dt)
    
    meshgrid = grid.meshgrid
    # Define Modules
    u = create_structured_warp_field(grid,'node',2)
    
    u_BC = GridBoundary(u,dx,ghost_cells)
    u_BC.dirichlet_BC('ALL',0.)
    u_BC.dirichlet_BC('+Y',1.,0)
    
    u_lapl = Laplacian(2,dx,ghost_cells)
    u_grad = Grad(2,u.shape,dx,ghost_cells=ghost_cells)
    
    u_outer = OuterProduct(2,2)
    u_conv_div = Divergence((2,2),u.shape,dx,ghost_cells)
    
    u_damping = Laplacian(2,dx,ghost_cells)
    
    rho = create_structured_warp_field(grid,'node',1)
    rho.fill_(1.)
    rho_BC = GridBoundary(rho,dx,ghost_cells)
    rho_BC.vonNeumann_BC('ALL',0.)
    p_grad = Grad(1,rho.shape,dx,ghost_cells)
    
    momentum_div = Divergence(2,u.shape,dx,ghost_cells)
    momentum = scalarVectorMult(2)

    @wp.func
    def get_p_op(rho:vector(1,float),c2:float,density:float):
        rho[0] = c2*(rho[0] - density)
        return rho
    
    get_p = MapWise(get_p_op)
    m_div_rho = MapWise(lambda m,rho: m/rho[0])
    t= 0
    
    def bc(t,m,rho):
        u = m_div_rho(m,rho)
        u_fix = u_BC(u)
        rho_fix = rho_BC(rho)
        m_fix = momentum(rho_fix,u_fix)
        return m_fix,rho_fix
    
    def f(t,m,rho):
        u = m_div_rho(m,rho)
        m_div = momentum_div(m,-1.)
        #Convec + diff
        u_diff = u_lapl(u,viscosity)        
        u_2 = u_outer(m,u)
        u_conv = u_conv_div(u_2)
        u_damp = u_damping(u_diff,damping_eps)
        #Pressure Gradient
        p = get_p(rho,c2,density)
        # print(p.numpy().squeeze())
        dp = p_grad(p,-1.)
        # Sum
        u_F = dp - u_conv + u_diff + u_damp
        return u_F,m_div

    RK2 = RungeKatta2([u.dtype,rho.dtype],f=f,bc= bc)
    m = momentum(rho,u)
    m,rho = RK2(t,dt,m,rho)
    
    with wp.ScopedCapture() as capture:
        m,rho = RK2(t,dt,m,rho)
    
    
    for i in range(10):
        wp.capture_launch(capture.graph)
        
    return True
# import pytest
# @pytest.fixture(scope="function")
def test_function():
    assert LBM_RK2()
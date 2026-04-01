from pde_module.LBM import BGK_collision,Streaming,Boundary
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.LBM import LBM_Mesh
from pde_module.mesh import to_pyvista
import pyvista as pv


wp.init()
# wp.config.mode = "debug"

def LBM_Channel_Flow():
    n = 11
    L,W = 1,3
    dx = L/(n-1)
    U_phs = 1.
    U = 0.025
    lbm = LBM_Mesh('D2Q9',dx,nodes_per_axis=(int(3*n),n,1),origin=(0.,0.,0.))
    # Runtime params
    viscosity = 1/100
    
    dt = dx*U/U_phs 
    Re = 1/viscosity
    
    L_lat = n
    v_lat = U*L_lat/Re
    tau = v_lat/(1/3.) +0.5
    print(tau)
    
    # Define Modules
    np.set_printoptions(precision=2)
    
    f_dist = lbm.create_field(9,0.1)
    rho = lbm.create_field(1)
    u = lbm.create_field(2)
    
    stream = Streaming.from_LBM_Mesh(lbm)
    collide = BGK_collision.from_LBM_Mesh(lbm)
    boundary = Boundary.from_LBM_Mesh(lbm)
    
    u_wall = (U,0.)
    boundary.set_BC('-X',3,velocity= u_wall)
    boundary.set_BC('+X',3,density=-1.)
    boundary.set_BC('+Y',1)
    boundary.set_BC('-Y',1)
    
    
    from warp.types import matrix
    @wp.kernel
    def calculate_rho_and_u(f_in:wp.array2d(dtype=float),rho:wp.array2d(dtype =float),u:wp.array2d(dtype =float),float_directions:matrix((9,2),dtype = float)):
        tid = wp.tid()
        
        rho_var = 0.
        u_var = wp.vec2f()
        for f in range(9):
            rho_var += f_in[f,tid]
            u_var += f_in[f,tid]*float_directions[f]
        rho[0,tid] = rho_var
        
        u_var/= rho_var
        
        u[0,tid] = u_var[0]/rho_var
        u[1,tid] = u_var[1]/rho_var
        
    for i in range(5):
            f_collide = collide(f_dist,tau)
            f_stream = stream(f_collide) 
            f_dist = boundary(f_stream)
        
    wp.launch(calculate_rho_and_u,dim = rho.size,inputs = [f_dist,rho,u,lbm.latticeModel.float_directions])
        
    return True


def test_function():
    assert LBM_Channel_Flow()


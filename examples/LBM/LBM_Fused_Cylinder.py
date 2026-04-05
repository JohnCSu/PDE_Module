import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.LBM import LBM_Mesh
from pde_module.LBM import FusedLBMKernel
from pde_module.mesh import to_pyvista
import pyvista as pv
import pandas as pd
from pde_module.visualization import Pyvista_Visualizer

wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 250
    L = 1
    
    k = 3
    W = k*L
    dx = L/(n-1)
    
    U_phs = 1.
    U = 0.1
    lbm = LBM_Mesh('D2Q9',dx,nodes_per_axis=(k*n,n,1),origin=(0.,0.,0.))
    # Runtime params
    viscosity = 1/500
    
    dt = dx*U/U_phs 
    Re = 1/viscosity
    
    L_lat = n
    v_lat = U*L_lat/Re
    tau = v_lat/(1/3.) +0.5
    print(f'Tau: {tau}, dt: {dt}')
    
    # Define Modules
    np.set_printoptions(precision=2)
    centre = (W/4,L/2)
    R = 0.1
    cyl = lambda x,y,z : (x-centre[0])**2 + (y-centre[1])**2 <= R**2
    lbm.from_bool_func(cyl)
    
    lbm_kernel = FusedLBMKernel.from_LBM_Mesh(lbm)
    
    lbm_kernel.sigma = 0.1
    u_wall = (U,0.)
    
    lbm_kernel.set_BC('-X',3,u_wall)
    lbm_kernel.set_BC('+X',3,density= -1.)
    lbm_kernel.set_BC('-Y',0)
    lbm_kernel.set_BC('+Y',0)
    
    from warp.types import matrix
    @wp.kernel
    def calculate_rho_and_u(f_in:wp.array2d(dtype=float),rho:wp.array2d(dtype =float),u:wp.array2d(dtype =float),float_directions:matrix((9,2),dtype = float)):
        tid = wp.tid()
        
        rho_var = 0.
        for f in range(9):
            rho_var += f_in[f,tid]
        rho[0,tid] = rho_var
        
        u_var = wp.vec2f()
        for f in range(9):
            u_var += f_in[f,tid]*float_directions[f]
            
        u[0,tid] = u_var[0]/rho_var
        u[1,tid] = u_var[1]/rho_var
        
    f_dist = lbm.create_field(9,0.1)
    rho = lbm.create_field(1)
    u = lbm.create_field(2)
        
    # print(boundary.flags.squeeze())
    u_np = u.numpy()/U
    
    v_benchmark = pd.read_csv(r'examples/v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples/u_velocity_results.txt',sep= '\t')
    
    Viewer = Pyvista_Visualizer(lbm,(2,1))
    Viewer.add_point_data( {'u_mag':np.sqrt(u_np[0]**2 + u_np[1]**2),
                            'U velocity':u_np[0],
                            'V velocity':u_np[1],
                            }
                          )
    
    Viewer.set_mesh_display('u_mag',cmap='jet',show_edges=False,clim = [0,1.5])

    Viewer.add_chart((1,0),'U velocity',(W/2,0,0),(W/2,L,0),1,resolution= n,label = 'LBM')
    
    Viewer.set_Animation('FusedLBM.gif')
    timer_pos = (0.5,0.8)
    Viewer.plotter.subplot(*(0,0))
    timer = Viewer.plotter.add_text("steps: 0, Time: 0.00 sec", position=timer_pos, font_size=12,viewport = True)
    num_frames = 1000
    step_per_frame = 200
    
    
    
    
    f_out = None
    T = 10.
    ramp_fn = lambda t: min(t/T,1.)
    
    for frame in range(num_frames):
        steps = frame*step_per_frame
        t = steps*dt
        
        lbm_kernel.ramp = ramp_fn(t)
        for i in range(step_per_frame):
            f_out = lbm_kernel(f_dist,tau,f_out = f_out)
            f_dist,f_out = f_out, f_dist
        
        wp.launch(calculate_rho_and_u,dim = rho.size,inputs = [f_out,rho,u,lbm.latticeModel.float_directions])
        
        u_np = u.numpy()/U
        Viewer.mesh.point_data['u_mag'] = np.sqrt(u_np[0]**2 + u_np[1]**2)
        Viewer.mesh.point_data['U velocity'] = u_np[0]
        Viewer.mesh.point_data['V velocity'] = u_np[1]
        timer.input = f"steps: {steps}, Time: {t:.2f} sec"
        Viewer.update_charts_and_write_frame()
    
    Viewer.close()
    
    
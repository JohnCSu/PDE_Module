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
    n = 101
    L = 1
    dx = L/(n-1)
    U_phs = 1.
    U = 0.05
    lbm = LBM_Mesh('D2Q9',dx,nodes_per_axis=(n,n,1),origin=(0.,0.,0.))
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
    shape = (9,n,n,1)
    f_dist = np.ones(shape,dtype= np.float32)/9

    flatten = f_dist.reshape((9,-1))
    f_dist = wp.array(flatten)
    
    lbm_kernel = FusedLBMKernel.from_LBM_Mesh(lbm)
    
    u_wall = (U,0.)
    lbm_kernel.set_BC('+Y',2,u_wall)
    lbm_kernel.set_BC('-X',1)
    lbm_kernel.set_BC('+X',1)
    lbm_kernel.set_BC('-Y',1)
    
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
        
        
    rho = lbm.create_field(1)
    u = lbm.create_field(2)
        
    # print(boundary.flags.squeeze())
    u_np = u.numpy()/U
    
    v_benchmark = pd.read_csv(r'examples/v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples/u_velocity_results.txt',sep= '\t')
    
    Viewer = Pyvista_Visualizer(lbm,(3,1))
    Viewer.add_point_data( {'u_mag':np.sqrt(u_np[0]**2 + u_np[1]**2),
                            'U velocity':u_np[0],
                            'V velocity':u_np[1],
                            }
                          )
    
    Viewer.set_mesh_display('u_mag')
    
    Viewer.add_chart((1,0),'V velocity',(0,L/2,0),(L,L/2,0),0,resolution= n,label = 'LBM')
    Viewer.add_data_to_chart((1,0),v_benchmark['%x'],v_benchmark['100'],color = 'r',label = 'Ghia et al')
    
    Viewer.add_chart((2,0),'U velocity',(L/2,0,0),(L/2,L,0),1,resolution= n,label = 'LBM')
    Viewer.add_data_to_chart((2,0),u_benchmark['%y'],u_benchmark['100'],color = 'r',label = 'Ghia et al')
    
    Viewer.set_Animation('FusedLBM.gif')
    
    num_frames = 300
    step_per_frame = 300
    
    f_out = None
    for frame in range(num_frames):
        t = frame*step_per_frame
        for i in range(step_per_frame):
            f_out = lbm_kernel(f_dist,tau,f_out)
            f_dist,f_out = f_out, f_dist
        
        wp.launch(calculate_rho_and_u,dim = rho.size,inputs = [f_out,rho,u,lbm.latticeModel.float_directions])
        
        u_np = u.numpy()/U
        Viewer.mesh.point_data['u_mag'] = np.sqrt(u_np[0]**2 + u_np[1]**2)
        Viewer.mesh.point_data['U velocity'] = u_np[0]
        Viewer.mesh.point_data['V velocity'] = u_np[1]
        Viewer.update_charts_and_write_frame()
    
    Viewer.close()
    
    
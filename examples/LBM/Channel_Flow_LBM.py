from pde_module.LBM import BGK_collision,Streaming,Boundary
import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.LBM import LBM_Mesh
from pde_module.mesh import to_pyvista
import pyvista as pv


wp.init()
# wp.config.mode = "debug"

if __name__ == '__main__':
    n = 201
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
    boundary.set_BC('-X',2,velocity= u_wall)
    boundary.set_BC('+X',3,density = 1.)
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
        
        
    
        
    # print(boundary.flags.squeeze())
    u_np = u.numpy()/U
    f_np = f_dist.numpy()[1]
    
    
    import pandas as pd
    v_benchmark = pd.read_csv(r'examples/v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples/u_velocity_results.txt',sep= '\t')
    
    pv_mesh = to_pyvista(lbm)
    pv_mesh.point_data['u_mag'] = np.sqrt(u_np[0]**2 + u_np[1]**2)
    pv_mesh.point_data['U velocity'] = u_np[0]
    pv_mesh.point_data['V velocity'] = u_np[1]


    plotter = pv.Plotter(shape = (2,1))
    
    plotter.subplot(0,0)
    plotter.add_mesh(pv_mesh, scalars = 'u_mag',show_edges = False, cmap= 'jet', clim = [0,1])
    plotter.view_xy()
    
    
    plotter.subplot(1,0)
    u_chart= pv.Chart2D()
    vertical_line = pv_mesh.sample_over_line((W/2,0,0),(W/2,L,0),resolution= n)
    u_05 = vertical_line.point_data['U velocity']
    u_line = u_chart.line(vertical_line.points[:,1],u_05,label = 'LBM')
    plotter.add_chart(u_chart)
    
    
    plotter.show(interactive_update=True)
    plotter.open_movie("transient_animation.gif")
    timer_pos = (0.5,0.8)
    timer = plotter.add_text("steps: 0", position=timer_pos, font_size=12,viewport = True)
    
    
    num_frames = 500
    step_per_frame = 300

    flags = wp.array(boundary.flags,dtype = wp.uint8)
    for frame in range(num_frames):
        t = frame*step_per_frame
        for i in range(step_per_frame):
            f_collide = collide(f_dist,tau)
            f_stream = stream(f_collide) 
            f_dist = boundary(f_stream)
        
        wp.launch(calculate_rho_and_u,dim = rho.size,inputs = [f_dist,rho,u,lbm.latticeModel.float_directions])
        u_np = u.numpy()/U
        pv_mesh.point_data['u_mag'] = np.sqrt(u_np[0]**2 + u_np[1]**2)
        pv_mesh.point_data['U velocity'] = u_np[0]
        pv_mesh.point_data['V velocity'] = u_np[1]
        timer.input = f"steps: {t}"
                
        vertical_line = pv_mesh.sample_over_line((W/2,0,0),(W/2,L,0),resolution= n)
        u_05 = vertical_line.point_data['U velocity']
        u_line.update(vertical_line.points[:,1],u_05)
        # print(u_np[0].reshape((n,n)).squeeze())
        
        
        plotter.write_frame()
    
    plotter.close()
    
    
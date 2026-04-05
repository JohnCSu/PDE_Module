import pyvista as pv
import numpy as np
from pde_module.mesh import to_pyvista,Mesh 
from typing import Optional
from dataclasses import dataclass

@dataclass
class Line:
    scalar: str
    p1:tuple[int]
    p2:tuple[int]
    axis:int
    label: str
    resolution:int
    data_type:str 
    pv_line: pv.charts.LinePlot2D

class Pyvista_Visualizer():
    plotter: pv.Plotter
    '''Pyvista Plotter class that controls all animation'''
    mesh:pv.UnstructuredGrid
    def __init__(self,mesh:Mesh,subplot_shape,**pyvista_Plotter_kwargs):
        self.subplot_shape =subplot_shape
        self.plotter = pv.Plotter(shape = (3,1),**pyvista_Plotter_kwargs)
        self.mesh = to_pyvista(mesh)
        self.charts:dict[tuple[int],pv.Chart2D] = {}
        self.lines:dict[tuple[int],dict[str,Line]] = {}
        
        

    def set_mesh_display(self,to_plot:str,**kwargs):
        plotter = self.plotter
        plotter.subplot(0,0)
        plotter.add_mesh(self.mesh, scalars = to_plot,**kwargs)
        plotter.view_xy()
    
    def add_point_data(self,point_data_dict:Optional[dict] = None,**kwargs):
        if isinstance(point_data_dict,dict):
            for key,val in point_data_dict.items():
                self.mesh.point_data[key] = val
        
            for key,val in kwargs.items():
                self.mesh.point_data[key] = val
                
    def add_cell_data(self,point_data_dict:Optional[dict] = None,**kwargs):
        if isinstance(point_data_dict,dict):
            for key,val in point_data_dict.items():
                self.mesh.cell_data[key] = val
        
            for key,val in kwargs.items():
                self.mesh.cell_data[key] = val
    
    
    def add_chart(self,subplot,scalar,p1,p2,axis,resolution,label,data_type = 'point'):
        
        self.plotter.subplot(*subplot)
        
        chart = pv.Chart2D()
         
        self.lines[subplot] = {}
        
        line = self.mesh.sample_over_line(p1,p2,resolution= resolution)
        
        if data_type == 'point':
            y = line.point_data[scalar]
        else:
            y = line.cell_data[scalar]
            data_type = 'cell'
        
        
        x = line.points[:,axis]
        line = chart.line(x,y,label = label)
        
        self.lines[subplot][label] = Line(scalar,p1,p2,axis,label,resolution,data_type,line)
        
        self.charts[subplot] = chart
        self.plotter.add_chart(self.charts[subplot])
        
    
    def add_data_to_chart(self,subplot,x,y,label,**kwargs):
        '''
        Add immutable data to a chart. This does not get updated. Use for Benchmark Data
        '''
        assert isinstance(self.charts[subplot],pv.Chart2D)
        self.charts[subplot].line(x,y,label = label,**kwargs)
         
    
    def set_Animation(self,name):
        self.plotter.show(interactive_update= True)
        self.plotter.open_movie(name)
        
    
    def update_charts_and_write_frame(self):
        for key in self.charts.keys():
            lines = self.lines[key]
            for label in lines.keys():
                line = lines[label]
                sample_line = self.mesh.sample_over_line(line.p1,line.p2,resolution=line.resolution)
                if line.data_type == 'point':
                    y = sample_line.point_data[line.scalar]
                else:
                    y = sample_line.cell_data[line.scalar]
                    
                x = sample_line.points[:,line.axis]
                line.pv_line.update(x,y)
        
        self.plotter.write_frame()
        
    
    def close(self):
        self.plotter.close()
                        
                
import warp as wp
import numpy as np
from math import prod
from warp.types import vector,matrix
class Grid:
    '''
    Class to create a Uniform Grid of spacing dx. nodal and cell fields can be created from this class
    '''
    dimension:int
    wp_float_dtype:wp.float32 | wp.float64 = wp.float32
    has_ghost_cells:bool = False
    
    @property
    def np_float_dtype(self):
        return wp.dtype_to_numpy(self.wp_float_dtype)
    
    @property
    def cell_volume(self):
        return self.dx**self.dimension
    @property
    def face_volume(self):
        return self.dx**(self.dimension-1) 
    
    @property
    def is_dimension(self):
        return (1 if p > 1 else 0 for p  in self.num_points )
    
    @property
    def dimension(self):
        return sum(self.is_dimension)
    
    @property
    def grid_types(self):
        return ('cell','node')
    
    @property
    def array_formats(self):
        return ('AoS','SoA')
    
    @property
    def array_types(self):
        return 'numpy','warp'
    
    def __init__(self,dx:float,num_points:tuple[int],origin = None,float_dtype = wp.float32):
        '''
        Create A grid object:
        Inputs:
        - dx: float spacing of each cube
        - num_points: tuple or iterable of 3 integers represnting the following:\n
            x: number of points along x (inlcuding origin)
            y: number of points along y (inlcuding origin)
            z: number of points along z (inlcuding origin)
        - origin : None or Tuple|Iterable of 3 floats: origin of cube. If None, then the default origin is (0,0,0)
        
        '''
        assert len(num_points) == 3
        assert all((isinstance(p,int) and p >= 1) for p in num_points),'All point in num points must be int and greater than 0'
        
        self.wp_float_dtype = float_dtype
        self.num_points = np.array(num_points,dtype=self.np_float_dtype)
        self.dx = self.np_float_dtype(dx)
        self.element_volume = dx**3
        self.face_area = dx**2
        self.origin = np.zeros(3,dtype=self.np_float_dtype) if origin is None else np.array(origin,dtype=self.np_float_dtype)
        self.nodal_coordinates_vectors = tuple(np.arange(0,axis,dtype=self.np_float_dtype)*dx - axis_origin for axis,axis_origin in zip(self.num_points,self.origin))
        self.cell_centroid_coordinate_vectors = tuple(
                                            (np.arange(0,n-1,dtype=self.np_float_dtype)*dx + dx/2 - axis_origin) if n > 1 
                                            else np.zeros(1,dtype=self.np_float_dtype) 
                                            for n,axis_origin in zip(self.num_points,self.origin) 
                                            )

        self.cell_grid_shape = (len(coord) for coord in self.cell_centroid_coordinate_vectors)
        self.node_grid_shape = (len(coord) for coord in self.cell_centroid_coordinate_vectors)
        
    def create_grid(self,grid_type:str,num_ghost_cells):
        assert grid_type in self.grid_types, 'Valid grid types are cell and node'
        
        if grid_type == 'cell':
            grid = np.meshgrid(*self.cell_centroid_coordinate_vectors)    
        elif grid_type == 'node':
            grid = np.meshgrid(*self.nodal_coordinates_vectors)
        return grid
    
    
    @staticmethod
    def _add_ghost_cells(grid_shape,num_ghost_cells):
        num_ghost_cells = tuple(num_ghost_cells for _ in range(3)) if isinstance(num_ghost_cells,int) else num_ghost_cells
        assert len(num_ghost_cells) == 3
        return (axis + g*2 for axis,g in zip(grid_shape,num_ghost_cells))
    
    def create_field(self,grid_type:str,num_fields:int |tuple[int],num_ghost_cells:int |tuple[int] ,array_type:str = 'warp',array_format:str = 'AoS') -> np.ndarray | wp.array:
        
        assert grid_type in self.grid_types
        assert array_type in self.array_types
        assert array_format in self.array_formats
        
        grid_shape = self.cell_grid_shape if grid_type == 'cell' else self.node_grid_shape
        grid_shape = self._add_ghost_cells(grid_shape,num_ghost_cells)
        
        if array_type == 'warp':
            if isinstance(num_fields,int):
                struct = vector(length=num_fields,dtype = self.wp_float_dtype)
            elif isinstance(num_fields,tuple):
                assert len(num_fields) == 2, 'num fields of tuple only support matrix'
                struct = matrix(shape =num_fields,dtype= self.wp_float_dtype)        
            return wp.zeros(shape=grid_shape,dtype=struct)
        
        elif array_type == 'numpy':
            if isinstance(num_fields,int):
                num_fields = (num_fields,)
            assert isinstance(num_fields,tuple), 'num fields must be int or tuple of ints'
            return np.zeros(shape = grid_shape + num_fields,dtype=self.np_float_dtype)
            
    
    def create_cell_field(self,num_fields:int |tuple[int],num_ghost_cells:int |tuple[int],array_type:str = 'warp',array_format:str = 'AoS'):
        return self.create_field('cell',num_fields,num_ghost_cells,array_type,array_format)
    
    def create_node_field(self,num_fields:int |tuple[int],num_ghost_cells:int |tuple[int],array_type:str = 'warp',array_format:str = 'AoS'):
        return self.create_field('node',num_fields,num_ghost_cells,array_type,array_format)
    
        
if __name__ == '__main__':
    
    grid = Grid(0.3,(5,5,1))

    
    meshgrid = grid.create_grid('cell',0)
    g = grid.create_field('cell',3,1)
    print(meshgrid[0].shape,g.shape)
    # print(meshgrid[0].shape)
    # print(grid.nodal_coordinates_vectors)
    # print(grid.cell_centroid_coordinate_vectors)

        
        
    
        
        
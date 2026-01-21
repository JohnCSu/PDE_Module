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
    
    def __init__(self,dx:float,num_points:tuple[int],origin = None,float_dtype = wp.float32,ghost_cells = None):
        '''
        Create A Uniform Grid.
        
        Note that the grid is always specified with 3 Coordinates (x,y,z) regardless of dimension. Grids of lower dimension
        will have a 1 in corresponding axes. For example 2D grid is shape (x,y,1)
        
        Parameters
        ----------
            dx (float) : grid spacing
            num_points (tuple[int]|Iterable[int]) : tuple or iterable of 3 integers represnting the following:
                1. x number of points along x (inlcuding origin)
                2. y number of points along y (inlcuding origin)
                3. z number of points along z (inlcuding origin)
            origin (None or (Tuple|Iterable)[float],optional) : origin of cube. If None, then the default origin is (0,0,0)
            float_dtype (wp.floatType) : the default float type for floating point arrays, default is wp.float32
            ghost cells (int | None, optional) : 
            int number of ghost cells to add to each axis that has more than one point. If ghost_cells is None, then the default value is 0
        Returns
        ---------
            None
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
                                            else np.ones(1,dtype=self.np_float_dtype)*axis_origin
                                            for n,axis_origin in zip(self.num_points,self.origin) 
                                            )

        self.cell_grid_shape = tuple(len(coord) for coord in self.cell_centroid_coordinate_vectors)
        self.node_grid_shape = self.num_points
        
        self.ghost_cells = 0 if ghost_cells is None else ghost_cells
        assert isinstance(self.ghost_cells,int) and self.ghost_cells >= 0 

    
    
    def get_coord_vectors(self,grid_type):
        assert grid_type in self.grid_types, 'Valid grid types are cell and node'
        return self.cell_centroid_coordinate_vectors if grid_type == 'cell' else self.nodal_coordinates_vectors
    
    
    def create_meshgrid(self,grid_type:str,ghost_cells = None,stack_axis = None,indexing = 'ij'):
        '''
        Create a meshgrid out of the coordinate vectors of the grid depending on grid type
        
        Parameters
        ----------
            grid_type (str) : 
                type of grid to generate can be cell or node
            ghost_cells (None | str,optional) :
                add ghost cells to grid. If None will use the grid's internal ghost_cells property
                
                Default is None
            stack_axis (None| int , optional) :
                whether to optionally stack the meshgrid arrays into a single array. If None, 
                then the standard meshgrid (list of arrays) is returned otherwise specify the
                stacking axis

                Default is None
            indexing ({'ij','xy'} str, optional) :
                indexing format, if 'ij' meshgrids are shape (x,y,z), if 'xy' shape is (y,x,z)
                Default is 'ij'    
        
        Returns
        --------
            grid (list[array] | array) : list of meshgrid arrays (same as np.meshgrid) or a single array of meshgrids arrays if stack_axes is not None
        
        '''
        
        coord_vectors = self.get_coord_vectors(grid_type)
        coord_vectors = self.add_ghost_coords(coord_vectors,ghost_cells)
        
        grid = np.meshgrid(*coord_vectors,indexing = indexing)
        if stack_axis is not None:
            assert isinstance(stack_axis,int)
            grid = np.stack(grid,axis = stack_axis)
        
        return grid
    
    
    def add_ghost_coords(self,coord_vectors,ghost_cells):
        ghost_cells = self._overide_ghost_cells(ghost_cells)
        if ghost_cells == 0:
            return coord_vectors
        # We need to add points dx
        coord_with_ghost = []
        for coord_vector in coord_vectors:
            if len(coord_vector) == 1: # If one point then leave alone
                coord_with_ghost.append(coord_vector)    
            else:
                left_g = np.array([coord_vector[0] - n*self.dx for n in range(1,ghost_cells+1)])
                right_g = np.array([coord_vector[-1] + n*self.dx for n in range(1,ghost_cells+1)])
                coord_with_ghost.append(np.concat((left_g,coord_vector,right_g),dtype=self.np_float_dtype))
                
        return coord_with_ghost
            
    
    def _overide_ghost_cells(self,ghost_cells):
        if ghost_cells is None:
            ghost_cells = self.ghost_cells
        assert isinstance(ghost_cells,int) and ghost_cells >= 0 
        return ghost_cells
    
    def _add_ghost_cells(self,grid_shape,ghost_cells):
        ghost_cells = self._overide_ghost_cells(ghost_cells)
        return tuple(axis + ghost_cells*2 if axis > 1 else axis for axis in (grid_shape))
    
    def create_field(self,grid_type:str,num_fields:int |tuple[int],ghost_cells:int|None = None ,array_type:str = 'warp',array_format:str = 'AoS') -> np.ndarray | wp.array:
        '''
        Create field array from grid. Currently only AoS arrays are supported
        
        Args:
            grid_type ({'cell','node'}, str) :
                type of grid to create field from
            num_fields (int|tuple[int]) :
                field shape to form. if integer or tuple of size 1, a vector dtype is infered. if tuple is of size 2, a matrix dtype is inferred
            ghost_cells (None | int,optional) :
                add ghost cells to grid. If None will use the grid's internal ghost_cells property

                Default is None
            array_type ({'warp','numpy'}, optional) :
                output array type of either warp or numpy

                Default is warp
            array_format ({'AoS','SoA'}, optional) :
                style of array format. Currently only AoS arrays are supported
        
        Returns:
            array (warp.array | numpy.array) :
                - if warp array, then the dtype of array is defined from num_fields and has shape of grid
                - if numpy array, then first 3 axes are the grid shape and trailing axes define the shape of structure
        '''
        assert grid_type in self.grid_types
        assert array_type in self.array_types
        assert array_format in self.array_formats
        
        grid_shape = self.cell_grid_shape if grid_type == 'cell' else self.node_grid_shape
        
        grid_shape = self._add_ghost_cells(grid_shape,ghost_cells)
        
        
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
            
    
    def create_cell_field(self,num_fields:int |tuple[int],ghost_cells:int |tuple[int],array_type:str = 'warp',array_format:str = 'AoS'):
        return self.create_field('cell',num_fields,ghost_cells,array_type,array_format)
    
    def create_node_field(self,num_fields:int |tuple[int],ghost_cells:int |tuple[int],array_type:str = 'warp',array_format:str = 'AoS'):
        return self.create_field('node',num_fields,ghost_cells,array_type,array_format)
    
    
    def initial_condition(self,grid_type,func,ghost_cells = None,array_type = 'warp',**kwargs):
        '''
        Use coordinates (x,y,z) generated from meshgrid and user specified function to generate initial conditions
        
        Args:
            grid_type ({'cell','node'}, str) :
                type of grid to create field from
            func (Callable[x,y,z]) :
                function that takes in a meshgrid output and outputs the initial condition. The first 3 axes
                should match the grid shape with the trailing axes used to determine the dtype of the field
            ghost_cells (None | int,optional) :
                add ghost cells to grid. If None will use the grid's internal ghost_cells property

                Default is None
            array_type ({'warp','numpy'}, optional) :
                output array type of either warp or numpy
            **kwargs (optional) :
                Any keyword arguments to pass to func

        Returns:
            array (warp.array | numpy.array) :
                - if warp array, then the dtype of array is defined from num_fields and has shape of grid
                - if numpy array, then first 3 axes are the grid shape and trailing axes define the shape of structure
    
        Example:
            ```python
                from matplotlib import pyplot as plt
                
                dx = 0.1 
                grid = Grid(dx,(11,1,1),ghost_cells= 1)
                
                meshgrid = grid.create_meshgrid('node') # Creates a list of (11,1,1) arrays
                f = lambda x,y,z: (x**2)[:,np.newaxis] # Output is (*grid_shape,1) i.e. (11,1,1)
                
                IC = grid.initial_condition('node',f) # This is same as calling grid.create_meshgrid and then f(*meshgrid)
                y = IC.numpy().squeeze() 
                
                plt.plot(meshgrid[0].squeeze(),y)
                plt.show()
            ```
        '''
        grid = self.create_meshgrid(grid_type,ghost_cells)
        output = func(*grid,**kwargs)
        
        grid_shape = self.cell_grid_shape if grid_type == 'cell' else self.node_grid_shape
        
        grid_shape = self._add_ghost_cells(grid_shape,ghost_cells)
        
        if grid_shape != output.shape[:3]:
            raise ValueError(f'grid shape of {grid_shape} was given but output shape first 3 axes was {output.shape[:3]}')
        
        assert array_type in self.array_types
        
        if array_type == 'warp':
            if len(output.shape) == 3: 
                output = output[:,:,:,np.newaxis]
            
            assert len(output.shape) <= 5, 'initial condiition only supports output arrays with at most 5 axes'
            if len(output.shape) == 4:
                return wp.array(output,shape = output.shape[:3],dtype=vector(output.shape[-1],dtype = self.wp_float_dtype))
            else:
                return wp.array(output,shape = output.shape[:3],dtype=matrix(output.shape[3:],dtype = self.wp_float_dtype))
            
        else:
            return output
        
        
if __name__ == '__main__':
    dx = 0.1 
    grid = Grid(dx,(11,1,1),ghost_cells= 1)

    
    meshgrid = grid.create_meshgrid('node')
    
    f = lambda x,y,z: (x**2)[:,np.newaxis]
    
    print(meshgrid[0].squeeze())
    IC = grid.initial_condition('cell',f)
    y = IC.numpy().squeeze()
    
    
    from matplotlib import pyplot as plt
    plt.plot(meshgrid[0].squeeze(),y)
    plt.show()
        
        
    
        
        
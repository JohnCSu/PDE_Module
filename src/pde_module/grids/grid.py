import warp as wp
import numpy as np
from math import prod

class Grid:
    '''
    Class to create a Uniform Grid of spacing dx. nodal and cell fields can be created from this class
    '''
    dimension:int
    wp_float_type:wp.float32 | wp.float64
    np_float_type:np.float32 | np.float64
    has_ghost_cells:bool = False
    
    
    def __init__(self,grid_type,dx,nx,ny=None,nz=None,origin = None,levels = None,warp_float_dtype = wp.float32):
        assert grid_type in {'cell','node'}
        self.grid_type = grid_type
        self.dx = dx
        self.nodal_shape = []
        self.cell_shape = []
        self.dimension = 0
        for n in [nx,ny,nz]:
            if isinstance(n,int):
                assert n > 1, 'number of nodes along a direction must be greater than 1. Use None for n = 1'
                self.nodal_shape.append(n)
                self.cell_shape.append(n-1)
                self.dimension += 1
            elif n is None:
                self.nodal_shape.append(1)
                self.cell_shape.append(1)
            else:
                raise ValueError()
        
        self.nodal_shape = tuple(self.nodal_shape)
        self.cell_shape = tuple(self.cell_shape)
        
        
        if origin is None:
            self.origin = (0.,)*3
        else:
            assert isinstance(origin,tuple|list)
            
            self.origin = origin
        
            if len(origin) == self.dimension:
                for _ in range(self.dimension,3):
                    self.origin += (0.,)
        
        self.cell_volume = dx**self.dimension
        # self.axis_ranges = [(o,dx*n) for o,n in zip(self.origin,self.shape)]
        # self.cell_centroids = tuple( [np.linspace(o+dx/2.,l[-1] - dx/2.,n) for o,l,n in zip(self.origin,self.axis_ranges,self.shape)])
        self.node_coordinate_arrays = tuple(np.linspace(o,o+self.dx*(n-1),n) for o,n in zip(self.origin,self.nodal_shape))
        self.cell_centroids = tuple( [np.linspace(o+dx/2.,l[-1]-dx/2.,n) if n > 1 else np.array([o]) for o,l,n in zip(self.origin,self.node_coordinate_arrays,self.cell_shape)])
        
        self.wp_float_type = warp_float_dtype
        self.np_float_type = wp.dtype_to_numpy(self.wp_float_type)
        
        self.levels = wp.vec3i(self.add_ghost_cells(levels))
        self.ghost_cell_shape = tuple(axis+2*level for axis,level in zip(self.cell_shape,self.levels))
        self.ghost_nodal_shape = tuple(axis+2*level for axis,level in zip(self.nodal_shape,self.levels))

        
    def add_ghost_cells(self,levels):
        if levels is None:
            return (0.,0.,0.)
            
        elif isinstance(levels,int):
            levels = (levels,)*self.dimension
            
        elif isinstance(levels,(tuple,list)):
            assert all(isinstance(x,int) for x in levels)
            assert len(levels) == self.dimension
            levels = levels
        else:
            raise ValueError()
        
        for _ in range(self.dimension,3):
            levels = levels + (0,)   
        self.has_ghost_cells = True
        return levels
        
        
    def create_grid_point_field(self,batch_size = None,output_type = 'warp',grid_type = None):
        
        if grid_type is None:
            grid_type = self.grid_type
    
        if grid_type == 'cell':
            coordinate_vectors = self.cell_centroids
        elif grid_type == 'node':
            coordinate_vectors = self.node_coordinate_arrays
        else:
            raise ValueError(f'Valid options are strings "cell" or "node" got {grid_type} instead')
        
        
        coordinate_vectors = self._create_ghost_coordinate_arrays(*coordinate_vectors,levels=self.levels)
        
        grid_point_field = np.meshgrid(*coordinate_vectors,indexing='ij')
        
        if batch_size is None:
            grid_point_field =  np.stack(grid_point_field,axis = -1)
        else:
            assert isinstance(batch_size,int)
            
            grid_point_field = np.stack(grid_point_field,axis = -1)
            grid_point_field = grid_point_field[np.newaxis,:,:,:,:]
            
            grid_point_field =  np.repeat(grid_point_field,repeats= batch_size,axis = 0)

        
        if output_type == 'warp':
            return wp.array(grid_point_field,dtype=wp.vec3)
        elif output_type == 'numpy':
            return grid_point_field
        else:
            raise ValueError(f'output_type must be string warp or numpy got {output_type} instead')
        
    def meshgrid(self,grid_type,add_ghost,for_plotting = False,indexing = 'xy'):
        '''
        return a list of meshgrid (with xy indexing) arrays of gridpoints
        '''
        if grid_type is None:
            grid_type = self.grid_type
    
        if grid_type == 'cell':
            coordinate_vectors = self.cell_centroids
        elif grid_type == 'node':
            coordinate_vectors = self.node_coordinate_arrays
        else:
            raise ValueError(f'Valid options are strings "cell" or "node" got {grid_type} instead')
        
        if add_ghost:
            coordinate_vectors = self._create_ghost_coordinate_arrays(*coordinate_vectors,levels=self.levels)
        
        if for_plotting:
            return np.meshgrid(*coordinate_vectors[:self.dimension],indexing=indexing)
         
        return np.meshgrid(*coordinate_vectors,indexing=indexing)
    
    def initial_condition(self,func,grid_type = None,**kwargs):
        
        meshgrid = self.meshgrid(grid_type,add_ghost=True)
        output = func(*meshgrid,**kwargs) # Outputs would be N,M,O
        
        if grid_type is None:
            grid_type = self.grid_type
    
        if grid_type == 'cell':
            shape_with_ghost = self.ghost_cell_shape
        elif grid_type == 'node':
            shape_with_ghost = self.ghost_nodal_shape
        else:
            raise ValueError(f'Valid options are strings "cell" or "node" got {grid_type} instead')
        
          
        if output.shape == shape_with_ghost: # We have scalar output
            output = np.expand_dims(output,axis = (0,-1)) # We add a batch + vector output shape
        elif len(output.shape) == len(shape_with_ghost) + 1: # WE have a vector output
            output = np.expand_dims(output,axis = (0,)) # We add a batch dim only
        
        else:
            raise NotImplementedError(f'function should only output either scalar or vector output across the grid shape')
        
        
        return wp.array(output,dtype =wp.vec(length = output.shape[-1],dtype = float),shape = output.shape[:-1])
        
    def create_field(self,grid_type,output_shape:int|tuple|list,batch_size:int,flatten:bool):
        
        if grid_type is None:
            grid_type = self.grid_type
        
        if grid_type == 'cell':
            arr_shape = (batch_size,) + self.ghost_cell_shape
        elif grid_type == 'node':
            arr_shape = (batch_size,) + self.ghost_nodal_shape
        else:
            raise ValueError()
        
        if isinstance(output_shape,int):
                output_shape = (output_shape,)
        
        assert isinstance(output_shape,(tuple,list))
        
        if len(output_shape) == 1:
            dtype = wp.vec(length = output_shape[0],dtype = self.wp_float_type)
        elif len(output_shape) == 2:
            dtype = wp.mat(shape = output_shape,dtype = self.wp_float_type)
        else:
            raise ValueError('output shape must be int or tuple or list of size 1 or 2')
            
            
        arr = wp.zeros(shape = arr_shape, dtype = dtype)
        
        
        return arr.reshape((-1,prod(arr_shape[1:]))) if flatten else arr
    

    def create_nodal_field(self,output_shape:int|tuple|list,batch_size:int = 1,flatten:bool = False):
        '''
        Create a field based on the nodal shape of the grid plus any ghost cells
        
        output_shape: int|tuple|list, int is equivalent to a tuple of len 1. If tuple len == 1, then a vector field is created, if tuple len == 2 then a matrix field is created
        batch_size: int number of different fields to generate
        flatten:  bool if true then flattens to make a (B,N) where B is the batchsize and N is the product of the field shape 
        '''
        return self.create_field('node',output_shape,batch_size,flatten)
    
    def create_cell_field(self,output_shape:int|tuple|list,batch_size:int = 1,flatten:bool = False):
        '''
        Create a field based on the cell shape of the grid plus any ghost cells
        
        output_shape: int|tuple|list, int is equivalent to a tuple of len 1. If tuple len == 1, then a vector field is created, if tuple len == 2 then a matrix field is created
        batch_size: int number of different fields to generate
        flatten:  bool if true then flattens to make a (B,N) where B is the batchsize and N is the product of the field shape 
        '''
        return self.create_field('cell',output_shape,batch_size,flatten)
    
    
    def create_faces_field(self,output_shape:int|tuple|list,batch_size:int,flatten:bool):
        pass
    
    
    @property
    def shape(self):
        '''
        Grid shape excluding any ghost cells.
        
        shape tuple is given based on the value of self.grid_type
        '''
        if self.grid_type == 'cell':
            return self.cell_shape
        elif self.grid_type == 'node':
            return self.nodal_shape
        
        raise ValueError('Somehow attribute grid_type was not cell or node')
        
    @property
    def ghost_shape(self):
        '''
        Grid shape INCLUDING any ghost cells.
        
        shape tuple is given based on the value of self.grid_type
        '''
        
        if self.grid_type == 'cell':
            return self.ghost_cell_shape
        elif self.grid_type == 'node':
            return self.ghost_nodal_shape
        
        raise ValueError('Somehow attribute grid_type was not cell or node')
        
    
    
    @property
    def boundary_node_indices(self):
        '''
        return a Nx3 array of all nodal indices located at the boundary. Done lazily as may not be needed
        '''
        if not hasattr(self,'_boundary_indices'):    
            boundaries = []
            for i,axis_name in enumerate(['X','Y','Z'][:self.dimension] ):
                coords = self.nodal_shape[i]
                axis_lim = [0,coords-1]
                for fixed_point in axis_lim:
                    
                    shape = list(self.nodal_shape)
                    shape[i] = 1
                    
                    indices = np.indices(shape,dtype = int)
                    indices = np.moveaxis(indices,0,-1).reshape(-1,3)
                    indices[:,i] += fixed_point
                    
                    boundaries.append(indices)
                    
            self._boundary_indices = np.unique(np.concatenate(boundaries,dtype= int),axis = 0).astype(np.int32)

        return self._boundary_indices
    
    
    def calculate_boundary_Faces(self):
        pass
    
    def calculate_internal_Faces(self):
        pass
    
    
    def trim_ghost_values(self,arr,convert_to_numpy = True):
        '''
        Trim ghost cells from array
        '''
        
        if convert_to_numpy:
            assert isinstance(arr,wp.array)
            arr = arr.numpy()
        else:
            assert isinstance(arr,np.ndarray), 'array must be numpy array if convert to numpy is False '
            
            
        slices = []
        for i,level in enumerate(self.levels):
            if level == 0:
                slices.append(0)
            else:
                start = level
                end = (arr.shape[1+i])-level
                slices.append(slice(start,end))
            
        return arr[:,slices[0],slices[1],slices[2]]

    @staticmethod
    def _create_ghost_coordinate_arrays(x:np.ndarray,y:np.ndarray,z:np.ndarray,levels:tuple):
        
        ghost_list = []
        for num_ghost_cells,axis_arr in zip(levels,[x,y,z]):
            
            if num_ghost_cells > 0:
                    
                # For LHS i.e. negative values
                dx_l = np.cumsum(np.diff(axis_arr[:num_ghost_cells+1]))
                ghost_l = (axis_arr[0]- dx_l)[::-1]
                
                
                # For RHS
                axis_arr_rev = axis_arr[::-1]
                dx_r = np.cumsum(np.diff(axis_arr_rev[:num_ghost_cells+1]))
                ghost_r = (axis_arr_rev[0] - dx_r)

                arr_with_ghost = np.concatenate((ghost_l,axis_arr,ghost_r),dtype=axis_arr.dtype)
                
            else:
                arr_with_ghost = axis_arr
                    
            ghost_list.append(arr_with_ghost)
        
        return ghost_list




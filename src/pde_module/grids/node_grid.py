import numpy as np
import warp as wp
from typing import Callable,Any
import matplotlib.pyplot as plt
from collections import deque


VALID_ARRAY_TYPES = {
    'numpy',
    'warp',
    'pytorch',
    'jax-numpy',
}


class Ghost_cells():
    
    @staticmethod
    def set_ghost_cells_per_axis(ghost_cells:int|tuple[int],dimension:int) -> wp.vec3i:
        
        if isinstance(ghost_cells,int):
            ghost_cells_per_axis = (ghost_cells,)*dimension
            
        elif isinstance(ghost_cells,tuple):
            assert all([isinstance(ghost_cell,int) for ghost_cell in ghost_cells]), 'tuple must contain only ints'
            ghost_cells_per_axis = ghost_cells
            assert len(ghost_cells) == dimension or len(ghost_cells) == 3,'Ghost_cells tuple must be same as dimension or of size 3'
            
        else:
            raise TypeError('ghost cells must be either int or tuple of ints') 
        
        if len(ghost_cells_per_axis) < 3:
            for i in range(3-dimension):
                ghost_cells_per_axis += (0,)

        assert len(ghost_cells_per_axis) == 3 
        
        ghost_vec = wp.vec(length = len(ghost_cells_per_axis),dtype = int)
        
        return ghost_vec(*ghost_cells_per_axis)
    
    
    @staticmethod
    def add_ghost_cells(x:np.ndarray,y:np.ndarray,z:np.ndarray,ghost_cells_per_axis:tuple):
    
        ghost_list = []
        for num_ghost_cells,axis_arr in zip(ghost_cells_per_axis,[x,y,z]):
            
            if num_ghost_cells > 0:
                    
                dx_l = np.cumsum(np.diff(axis_arr[:num_ghost_cells+1]))
                
                ghost_l = (axis_arr[0]- dx_l)[::-1]
                
                axis_arr_rev = axis_arr[::-1]
                dx_r = np.cumsum(np.diff(axis_arr_rev[:num_ghost_cells+1]))
                ghost_r = (axis_arr[0] - dx_r)

                arr_with_ghost = np.concatenate((ghost_l,axis_arr,ghost_r),dtype=axis_arr.dtype)
                
            else:
                arr_with_ghost = axis_arr
                    
            ghost_list.append(arr_with_ghost)
        
        return ghost_list

    @staticmethod
    def create_stencil_grid(*coordinate_arrays:np.ndarray):
        max_length = max([len(arr) for arr in coordinate_arrays])
        arrs = []
        for arr in coordinate_arrays:
            pad = max_length - len(arr)
            arrs.append(np.pad(arr,(0,pad)))
        return np.stack(arrs,axis = 0)
    


class NodeGrid():
    '''
    Create a Grid of nodes for finite difference of shape (B,X,Y,Z)
    
    B: batch number
    X: number of points along x axis
    Y: number of points along y axis
    Z: number of points along z axis
    
    Leaving y or z as None sets that axis length to 1 (effectively reducing dimension by 1)
    '''
    is_warp:bool = False
    _meshgrid:None | np.ndarray = None
    def __init__(self,x,y=None,z=None,levels = 1,float_dtype = np.float32):
        assert isinstance(x,np.ndarray)
        self.wp_float_dtype = wp.dtype_from_numpy(float_dtype)
        
        arrs = []
        for axis in [x,y,z]:
            if axis is None:
                arr = np.array([0],dtype=float_dtype)
            else:
                arr = axis
                assert len(arr.shape) == 1 and len(arr) >=3, 'There must be at least 3 points along a specified axis'
            
            arrs.append(arr) 
                
        
        self.dimension = sum([1 for i in arrs if len(i) > 1])
        
        self.float_dtype = float_dtype
        
        
        self.coordinate_vectors = arrs     
        '''tuple of coordinate vectors along each axis that define the grid excluding ghost cells'''
        
        self.x,self.y,self.z = self.coordinate_vectors
        
        self.shape = tuple(len(arr) for arr in self.coordinate_vectors)
        '''Shape of grid excluding any ghost points'''
        
        
        self.levels = Ghost_cells.set_ghost_cells_per_axis(levels,self.dimension)
        self.stencil_coordinate_vectors = Ghost_cells.add_ghost_cells(*self.coordinate_vectors,ghost_cells_per_axis=self.levels)
        self.stencil_shape = tuple(len(arr) for arr in self.stencil_coordinate_vectors)
        '''Shape including ghost nodes'''
        self.stencil_points = Ghost_cells.create_stencil_grid(*self.stencil_coordinate_vectors)
        '''Stacked and padded arrays of (3,N) where N is the maximum number of points along each axis, this is passed into the stencil kernel to provide information about grid'''
        
        
        
        
        
        
    
    @property
    def meshgrid(self):
        '''Meshgrid value from coordinate vectors'''
        
        #We Ignore any axis
        if self._meshgrid is None:
            self._meshgrid =  np.meshgrid(*self.coordinate_vectors,indexing='ij')
            # self._plt_meshgrid = np.stack(meshgrid,axis = -1) 
        
        return self._meshgrid
    
    @property
    def plt_meshgrid(self):
        if hasattr(self,'_plt_meshgrid') is False:
            self._plt_meshgrid = np.meshgrid(*self.coordinate_vectors[0:self.dimension]) 
        return self._plt_meshgrid
    
    @property
    def meshgrid_with_ghost(self):
        if hasattr(self,'_meshgrid_with_ghost') is False:
            self._meshgrid_with_ghost= np.meshgrid(*self.stencil_coordinate_vectors,indexing= 'ij')
            
        return self._meshgrid_with_ghost
    
    def initial_condition(self,func:Callable,**kwargs):
        '''
        Take a scalar function of f(x,y,z) that returns an output vector or scalar.
        
        The function is then mapped over the meshgrid values 
        
        For lower dimensions you still need to have y and z defined which can simply not be used in the function or add *args to the function
        
        In the output we then add axis such that the output is of size (B,X,Y,Z,O) where
        B = 1 (batch axis)
        X,Y,Z = grid shape
        O = output axis (O=1 for scalar IC)
        
        '''
        
        output = func(*self.meshgrid_with_ghost,**kwargs)
        shape = self.stencil_shape
        if output.shape == shape: # We have scalar output
            output = np.expand_dims(output,axis = (0,-1)) # We add a batch + vector output shape
        elif output.shape[:-1] == shape and len(output.shape) == len(self.shape) + 1: # WE have a vector output
            output = np.expand_dims(output,axis = (0,)) # We add a batch 
        
        else:
            raise NotImplementedError(f'function should only output either scalar or 1D array output got {self.shape[3:]} instead')
        
        
        
        
        return wp.array(output,dtype =wp.vec(length = output.shape[-1],dtype = float),shape = output.shape[:-1])
    
    
    def create_grid_with_ghost(self,num_outputs,arr_type = 'warp'):
        shape = self.stencil_shape
        
        if arr_type == 'numpy':     
            return np.zeros(shape= shape + (num_outputs,),dtype = self.float_dtype)
        elif arr_type == 'warp':
            return wp.zeros(shape = shape, dtype = wp.vec(length = num_outputs,dtype = self.wp_float_dtype))
        else:
            raise ValueError()
        
        
    
    def trim_ghost_values(self,arr):
        '''
        Trim ghost cells from array
        '''
        x = arr
        slices = []
        for i,level in enumerate(self.levels):
            if level == 0:
                slices.append(0)
            else:
                start = level
                end = (arr.shape[1+i])-level
                slices.append(slice(start,end))
            
        return arr[:,slices[0],slices[1],slices[2]]

    
    def to_warp(self):
        self.is_warp = True
        
        # self.grid = wp.array(self.grid,dtype = wp.vec(length = 3,dtype=self.wp_float_dtype))
        # self.values = [wp.from_numpy(val,shape = val.shape[:-1],dtype = wp.vec(length = self.num_outputs,dtype=self.wp_float_dtype)) for val in self.values]
        
        # self.levels = wp.array(self.levels,dtype= int)
        self.stencil_points = wp.array(self.stencil_points,dtype=self.wp_float_dtype)
        
        
        return None
    def to_numpy(self):
        self.is_warp = False
        # self.levels = self.levels.numpy()
        self.stencil_points = self.stencil_points.numpy()
        
    
    # def to_warp(self,dt):
    #     if not self.is_warp:
    #         self.convert_to_warp()
    #     return [self.grid,self.values[0],self.values[1],dt]
    

if __name__ == '__main__':
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,10)
    a = NodeGrid(x,y,levels = (2,1))
    print(a.stencil_points.shape)
    print(a.levels)
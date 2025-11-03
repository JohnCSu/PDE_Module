import warp as wp
import numpy as np
# class UniformCellGrid():
#     '''
#     Uniform Grid Cell or voxel grid cell
    
#     nx: number of cells along x direction
#     ny: number of cells along y direction
#     nz: number of cells along z direction
    
#     Note this always creates 3D
#     '''
#     def __init__(self,dx,nx,ny,nz,origin = (0.,0.,0.),float_dtype = np.float32):
        
#         self.dx = dx
#         self.shape = (nx,ny,nz)
#         '''
#         Number of cells in each direction
#         '''
#         self.origin = origin
#         self.volume = dx**3.
        
#         self.dimension = 3
        
#         self.float_dtype = float_dtype
        
#         self.axis_lengths = [(o,dx*n) for o,n in zip(self.origin,self.shape)]
        
#         self.cell_centroids = tuple(np.linspace(o+dx/2.,l - dx/2.,n) for o,l,n in zip(self.origin,self.axis_lengths,self.shape))
    
    
    
            
        

import warp as wp
import numpy as np
class UniformCellGrid():
    '''
    Uniform Grid Cell or voxel grid cell
    
    nx: number of cells along x direction
    ny: number of cells along y direction
    nz: number of cells along z direction
    
    Note this always creates 3D
    '''
    def __init__(self,dx,nx,ny,origin = (0.,0.),float_dtype = np.float32):
        
        self.dx = dx
        self.shape = (nx,ny)
        
        '''
        Number of cells in each direction
        '''
        self.origin = origin
        self.dimension = len(self.shape)
        self.volume = dx**self.dimension
        self.float_dtype = float_dtype
        self.axis_ranges = [(o,dx*n) for o,n in zip(self.origin,self.shape)]
        self.cell_centroids = tuple( [np.linspace(o+dx/2.,l[-1] - dx/2.,n) for o,l,n in zip(self.origin,self.axis_ranges,self.shape)])
    
    
    def LBM(self,lattice_model = 'D2Q9'):
        assert lattice_model in {'D2Q9'}
        
        self.num_discrete_velocities = 9
        
        self.D2Q9_velocities = np.array([
            [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
            [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
        ]).swapaxes(0,1)
        self.velocity_directions_int =wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = int)(self.D2Q9_velocities)
        
        self.velocity_directions_float = wp.mat(shape = (self.num_discrete_velocities,self.dimension),dtype = float)(self.D2Q9_velocities.astype( self.float_dtype))
        
        
        self.indices = wp.vec(length = self.num_discrete_velocities,dtype = int)(np.arange(self.num_discrete_velocities,dtype= int))
        
        self.weights = wp.vec(length = self.num_discrete_velocities,dtype = wp.dtype_from_numpy(self.float_dtype) )([
                                4/9,                        # Center Velocity [0,]
                                1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
                                1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
                                ])
        
        self.latticeID = lattice_model
    
    def batch_shape(self,batch_size):
        return (batch_size,) + self.shape
    
    def generate_distribution_field(self,batch_size = 1):
        field_shape = (batch_size,) + self.shape
        return wp.zeros(shape = field_shape,dtype=wp.vec(self.num_discrete_velocities,dtype = float))
    def generate_velocity_field(self,batch_size = 1):
        field_shape = (batch_size,) + self.shape
        return wp.zeros(shape = field_shape,dtype=wp.vec(self.dimension,dtype = float))
    def generate_density_field(self,batch_size = 1):
        field_shape = (batch_size,) + self.shape
        return wp.zeros(shape = field_shape,dtype=wp.vec(1,dtype = float))
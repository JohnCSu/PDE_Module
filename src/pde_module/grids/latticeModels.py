import warp as wp
import numpy as np
from typing import Any


class LatticeUnits:
    cs: float = 1/(3.**0.5)
    
    def __init__(self,dx,density,dynamic_viscosity,u_ref,u_target = 0.1):
        self.dx = dx
        
        self.u_ref = u_ref
        self.u_target = u_target
        self.density = density
        self.dynamic_viscosity = dynamic_viscosity
        self.kinematic_viscosity = self.dynamic_viscosity/self.density
        
        self.cL = self.dx
        self.cU = self.u_ref/self.u_target
        self.cT = self.cL/self.cU
        self.cMu = self.cU*self.cL
        self.cRho = density
        
        
        self.dt = self.cT
        self.mu_lattice = self.kinematic_viscosity/self.cMu
        self.relaxation_factor = 3*self.mu_lattice+ 0.5
        
        self.Ma = u_target/self.cs
        
        assert self.Ma < 0.3, 'the lattice Mach number should be less than 0.3 ideally less than 0.1'
        assert 0.5 < self.relaxation_factor, 'relaxation factor must be greater than 0.5 for stability'
        
    


class LatticeModel:
    dimension:int
    float_dtype: Any
    num_discrete_velocities:int
    velocity_directions_int:wp.mat
    velocity_directions_float:wp.mat
    weights:wp.vec
    opposite_indices:wp.vec
    into_wall_indices:wp.mat
    



class D2Q9_Model:
    def __init__(self):
        self.dimension = 2
        self.float_dtype = np.float32
        
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

        self.opposite_indices = wp.vec(length = self.num_discrete_velocities,dtype = wp.int32)([
            0, 3, 4, 1, 2, 7, 8, 5, 6,
            ])
        
        self.into_wall_indices = wp.mat(shape = (4,3),dtype = int)([
        [3, 6, 7], #Left
        [1, 5, 8], #Right
        [2, 5, 6], #Up
        [4, 7, 8], #down
        ])
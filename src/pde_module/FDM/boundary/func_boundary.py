import warp as wp
import numpy as np
from typing import Any
from warp.types import vector,matrix,type_is_vector,type_is_float,is_float
from pde_module.utils.constants import Boundary_Types

def get_constant_func(constant,float_type):
    assert type_is_float(float_type) and is_float(constant)
    @wp.func
    def constant_value(
            current_values:wp.array3d(dtype = float_type),
            nodeID:wp.vec3i,
            varID:int,
            coordinates:wp.array3d(dtype=vector(3,float_type)),
            t:float,
            dx:float,
            params:Any):
        return constant
    
    return constant_value

from dataclasses import dataclass

@dataclass
class FunctionBC:
    face_ids:np.ndarray|wp.array
    boundary_type: np.int8 | wp.int8
    varID: int
    func: wp.Function
    kernel: wp.Kernel | None = None
    
    def to_warp(self):
        self.face_ids = wp.array(self.face_ids,dtype=int)
        self.boundary_type = wp.int8(self.boundary_type)
        
    def create_kernel(self,input_dtype,dx):
        self.kernel = create_func_boundary_kernel(self.func,self.boundary_type,self.varID,input_dtype,dx)
    



def create_func_boundary_kernel(func,boundary_type,varID,input_dtype,dx):
    boundary_type = int(boundary_type)
    assert boundary_type in Boundary_Types and boundary_type != Boundary_Types.NO_BC
    float_type = input_dtype._wp_scalar_type_
    dx = float_type(dx)
    
    @wp.kernel
    def func_boundary( current_values:wp.array3d(dtype = input_dtype),
                        ids:wp.array(dtype=int),
                        boundary_group_ijk_indices:wp.array(dtype=wp.vec3i),
                        boundary_interior:wp.array(dtype=wp.vec3i),
                        coordinates:wp.array3d(dtype=vector(3,float_type)),
                        t:wp.array(dtype=float_type),
                        params:Any,
                        new_values:wp.array3d(dtype = input_dtype)):
        tid = wp.tid()
        i = ids[tid]
        nodeID = boundary_group_ijk_indices[i]
        x = nodeID[0]
        y = nodeID[1]
        z = nodeID[2]
        interior_vec = boundary_interior[i]
        
        val = func(current_values,nodeID,varID,coordinates,t[0],dx,params) # Same shape or float? let make it float for simplicity
        # wp.printf('%f',val)
        for axis in range(3):
            if interior_vec[axis] != 0:                    
                inc_vec = wp.vec3i()
                inc_vec[axis] = interior_vec[axis]
                ghostID = nodeID - inc_vec
                adjID = nodeID + inc_vec
                
                if wp.static(boundary_type== Boundary_Types.DIRICHLET): 
                    new_values[x,y,z][varID] =  val
                    new_values[ghostID[0],ghostID[1],ghostID[2]][varID] =  float_type(2.)*val - current_values[adjID[0],adjID[1],adjID[2]][varID]
                elif wp.static(boundary_type== Boundary_Types.VON_NEUMANN):
                    new_values[ghostID[0],ghostID[1],ghostID[2]][varID] = - wp.sign(float_type(inc_vec[axis]))*float_type(2.)*dx*val + current_values[adjID[0],adjID[1],adjID[2]][varID]
        
    return func_boundary



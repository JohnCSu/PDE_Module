from pde_module.stencil import Stencil
from pde_module.stencil.hooks import *
from pde_module.utils import get_unique_key
import numpy as np
import warp as wp
from .finiteVolume import FiniteVolume
from ..flags import DIRICHLET,VON_NEUMANN
from ..mesh import FiniteVolumeMesh
from warp.types import vector


class Boundary(FiniteVolume):
    # For now lets just do fixed vonneuman and dirichlet
    
    def __init__(self,num_vars,mesh:FiniteVolumeMesh,float_dtype=wp.float32):
        assert isinstance(mesh,FiniteVolumeMesh)
        super().__init__(float_dtype)
        self.num_vars = num_vars
        self.num_faces = len(mesh.exterior_faces)
        self.mesh = mesh
        self.boundary_types = np.zeros((num_vars,self.num_faces),dtype = np.uint8)
        self.boundary_values = np.zeros((num_vars,self.num_faces),dtype= wp.dtype_to_numpy(float_dtype))
        self.groups = mesh.exterior_faces.groups
    
    def set_BC(self,ids,boundary_type,boundary_value,output_ids =None,boundary_group_name = None):
        match ids:
            case str():
                assert ids in self.groups.keys()
                groupName = ids
                ids = self.groups[ids]
            case _:
                groupName = get_unique_key(self.groups,base_name = 'BC_group') if boundary_group_name is None else boundary_group_name
                ids = np.array(ids,dtype= int)
                self.groups[groupName] = ids
        
        assert isinstance(groupName,str)    
        
        if output_ids is None:
            output_ids = slice(None)
            
        assert boundary_type in [DIRICHLET,VON_NEUMANN]
        
        self.boundary_types[output_ids,ids] = boundary_type
        self.boundary_values[output_ids,ids] = boundary_value                
    
    
    
    @setup
    def setup(self,cell_field):
        self.mesh.to_warp()
        self.boundary_types = wp.array(self.boundary_types,shape = self.boundary_types.shape)
        self.boundary_values = wp.array(self.boundary_values,shape = self.boundary_values.shape)
        self.kernel = create_FV_Boundary_kernel(self.num_vars,self.float_dtype)
        self.boundary_field = wp.empty_like(self.boundary_values,self.device)
        
            
    def forward(self,cell_field):
        exterior = self.mesh.exterior_faces
        
        wp.launch(kernel=self.kernel,
                  dim = self.num_faces,
                  inputs=[
                      cell_field,
                      exterior.cell_ids,
                      self.boundary_values,
                      self.boundary_types,
                      exterior.centroids,
                      self.mesh.cell_centroids
                  ],
                  
                  outputs = [self.boundary_field])
        
        
        return self.boundary_field
    
            


def create_FV_Boundary_kernel(num_vars,float_dtype):
    
    
    vec3 = vector(3,float_dtype)
    
    @wp.kernel
    def boundary_kernel(
        cell_field:wp.array2d[float_dtype],
        face_to_cell_id:wp.array1d[int],
        boundary_values:wp.array2d[float_dtype],
        boundary_types:wp.array2d[wp.uint8],
        exterior_face_centroid:wp.array1d[vec3],
        cell_centroids:wp.array1d[vec3],
        boundary_face_field:wp.array2d[float_dtype]
        ):
        
        tid = wp.tid() # Loop Face Ids of exterior
        
        cell_id = face_to_cell_id[tid]
        
        for var in range(num_vars):
            BC_type=  boundary_types[var,tid]
            BC_value = boundary_values[var,tid]
            if BC_type == DIRICHLET: # we can just set the 
                boundary_face_field[var,tid] = BC_value    
            else:
                boundary_face_field[var,tid] = cell_field[var,cell_id] + BC_value/wp.length(exterior_face_centroid[tid] - cell_centroids[tid])     
            
    return boundary_kernel
    
    

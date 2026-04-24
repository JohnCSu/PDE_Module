from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp
from warp.types import vector

from pde_module.stencil.hooks import *

class Diffusion(FiniteVolume):
    def __init__(self,mesh:FiniteVolumeMesh,interpolation = 'central',float_dtype = wp.float32):
        super().__init__(mesh,float_dtype)
        self.mesh = mesh
        self.interpolation_key = interpolation
    
    @setup
    def setup(self,input_field,boundary_values,viscosity):
        self.mesh.to_warp()
        self.field_shape = input_field.shape
        self.num_vars = self.field_shape[0]
        self.internal_kernel,self.external_kernel = create_diffusion_kernel(self.num_vars,self.float_dtype)
        
        self.output_field = wp.empty_like(input_field)
    
    
    def forward(self,input_field,boundary_values,viscosity):
        wp.launch(self.internal_kernel,dim = len(self.mesh.neighbors_offset),
                  inputs=[
                            input_field,
                            self.mesh.neighbors,
                            self.mesh.neighbors_offset,
                            self.mesh.face_normals,
                            self.mesh.cell_centroids,
                            self.mesh.cell_volumes,
                            viscosity,
                        ],
                  outputs=[
                            self.output_field        
                  ])
        
        
        wp.launch(self.external_kernel,dim = len(self.mesh.exterior_faces),
                  inputs=[
                            input_field,
                            self.mesh.exterior_faces.cell_ids,
                            boundary_values,
                            self.mesh.exterior_faces.centroids,
                            self.mesh.exterior_faces.normals,
                            self.mesh.cell_centroids,
                            self.mesh.cell_volumes,
                            viscosity,
                        ],
                  outputs=[
                            self.output_field        
                  ])
        
        
        return self.output_field
        
    
def create_diffusion_kernel(num_vars,float_dtype):
    vec_type= vector(num_vars,dtype = float_dtype)
    
    vector_type = vector(3,dtype = float_dtype)
    
    @wp.kernel
    def Internal_Face_Diffusion_kernel(
                        cell_values:wp.array2d[float_dtype],
                        cell_neighbors:wp.array1d[wp.vec2i],
                        cell_neighbors_offsets:wp.array1d[int],
                        face_normals:wp.array[vector_type],
                        cell_centroids:wp.array[vector_type],
                        cell_volumes:wp.array[float_dtype],
                        viscosity:float_dtype,
                        output_values:wp.array2d[float_dtype]
                        ):
        cell_id = wp.tid() # Loop through cells
        volume = cell_volumes[cell_id]
        
        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        
        dif = vec_type()
        for i in range(num_neighbors):
            # For now Assume orthogonal
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[-1]
            area = wp.length(face_normals[face_id])
            dist = wp.length(cell_centroids[neighbor_id] - cell_centroids[cell_id])
            for j in range(num_vars):
                val = (cell_values[j,neighbor_id] - cell_values[j,cell_id])*area/dist
                # wp.printf('%.2f \n',val)
                dif[j] += val
        # volume = 1.
        dif *= viscosity/volume
        for j in range(num_vars):
            output_values[j,cell_id] = dif[j]
        
    @wp.kernel
    def Boundary_Face_Diffusion_kernel(cell_values:wp.array2d[float_dtype],
                        boundary_face_to_cell_id:wp.array1d[int],
                        boundary_value:wp.array2d[float_dtype],
                        face_centroids:wp.array[vector_type],
                        face_normals:wp.array[vector_type],
                        cell_centroids:wp.array[vector_type],
                        cell_volumes:wp.array[float_dtype],
                         viscosity:float_dtype,
                         output_values:wp.array2d[float_dtype]):
        # Boudary Faces are guranted to be attached to only one cell so no race conditions or atomics needed
        
        tid = wp.tid() # Loop through Boundary Faces!
        
        cell_id = boundary_face_to_cell_id[tid]
        volume = cell_volumes[cell_id]
        dist = wp.length(face_centroids[tid] - cell_centroids[cell_id])
        area = wp.length(face_normals[tid])
        # volume = 1.
        alpha = viscosity*area/(volume*dist)
        
        for j in range(num_vars): 
            output_values[j,cell_id] += alpha*(boundary_value[j,tid] - cell_values[j,cell_id])
            
        
    return Internal_Face_Diffusion_kernel,Boundary_Face_Diffusion_kernel
            
        
        
        
        
        
        
    
    
    
    
    
    
    
    
        
        
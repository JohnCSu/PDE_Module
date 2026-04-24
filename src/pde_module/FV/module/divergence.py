from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp
from warp.types import vector

from pde_module.stencil.hooks import *

class Divergence(FiniteVolume):
    '''
    Only Vectors of size 3 is currently supported
    '''
    def __init__(self, mesh, float_dtype=wp.float32):
        super().__init__(mesh, float_dtype)
    @setup
    def initialise(self,vector_field,boundary_values,alpha =1.):
        self.field_shape = vector_field.shape
        assert self.field_shape[0] == 3 , 'Only vector fields of size 3 availiable for now'
        assert self.field_shape[0] == boundary_values.shape[0]
        self.num_vars = self.field_shape[0]
        self.num_cells = self.field_shape[-1]
        self.num_boundary_faces = len(self.mesh.exterior_faces)
        self.internal_kernel,self.external_kernel = create_Divergence_kernel(self.float_dtype)

        self.output_field = wp.empty((1,self.num_cells),dtype=self.float_dtype)
    def forward(self,vector_field,boundary_values,alpha =1.):
        wp.launch(self.internal_kernel,dim = self.num_cells,
                  inputs=[
                        vector_field,
                        alpha,
                        self.mesh.neighbors,
                        self.mesh.neighbors_offset,
                        self.mesh.face_normals,
                        self.mesh.face_centroids,
                        self.mesh.cell_centroids,
                        self.mesh.cell_volumes
                    ],
                  outputs =[
                      self.output_field
                  ])
        
        wp.launch(self.external_kernel,dim = self.num_boundary_faces,
                  inputs=[
                        boundary_values,
                        alpha,
                        self.mesh.exterior_faces.cell_ids,
                        self.mesh.exterior_faces.normals,
                        self.mesh.face_centroids,
                        self.mesh.cell_centroids,
                        self.mesh.cell_volumes
                  ],
                  outputs =[
                      self.output_field
                  ])
        
        return self.output_field
        

def create_Divergence_kernel(float_dtype):
    
    dimension = 3
    
    vec3 = vector(dimension,float_dtype)
    @wp.kernel
    def internal_divergence_kernel(
                                vector_field:wp.array2d[float_dtype],
                                alpha:float,
                                cell_neighbors:wp.array1d[wp.vec2i],
                                cell_neighbors_offsets:wp.array1d[int],
                                face_normals:wp.array1d[vec3],
                                face_centroids:wp.array1d[vec3],
                                cell_centroids:wp.array1d[vec3],
                                cell_volumes:wp.array1d[float_dtype],
                                scalar_field:wp.array2d[float_dtype]
                                   ):
        
        cell_id = wp.tid()
        cell_volume = cell_volumes[cell_id]
        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        cell_volume = cell_volumes[cell_id]

        div = float_dtype(0.)
        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[1]
            
            face_normal = face_normals[face_id]*wp.where(cell_id < neighbor_id,float_dtype(1.),float_dtype(-1.))
            
            for axis in range(dimension):
                div += ((vector_field[axis,cell_id] + vector_field[axis,neighbor_id])/2.) * face_normal[axis]
                
        scalar_field[0,cell_id] = div*alpha/cell_volume
    
    
    @wp.kernel
    def external_divergence_kernel( 
                                boundary_value:wp.array2d[float_dtype],
                                alpha:float_dtype,
                                boundary_face_to_cell_id:wp.array1d[int],
                                face_normals:wp.array1d[vec3],
                                face_centroids:wp.array1d[vec3], # Exterior
                                cell_centroids:wp.array1d[vec3], 
                                cell_volumes:wp.array1d[float_dtype],
                                scalar_field:wp.array2d[float_dtype]
                                ):
        
        tid = wp.tid()
        
        cell_id = boundary_face_to_cell_id[tid]
        face_normal = face_normals[tid]
        cell_volume = cell_volumes[cell_id]
        
        div = float_dtype(0.)
        for axis in range(dimension):
            div += boundary_value[axis,tid]*face_normal[axis]
            
        scalar_field[0,cell_id] = div*alpha/cell_volume 


    return internal_divergence_kernel,external_divergence_kernel
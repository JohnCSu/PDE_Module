from pde_module.FV.mesh import FiniteVolumeMesh
from .finiteVolume import FiniteVolume
import warp as wp
from warp.types import vector

from pde_module.stencil.hooks import *

class Advection(FiniteVolume):
    def __init__(self, mesh,interpolation = 'upwind', float_dtype=wp.float32):
        super().__init__(mesh, float_dtype)
        self.interpolation = interpolation
        
    
    def __call__(self,scalar_cell_field,scalar_boundary_values,velocity_cell_field,velocity_boundary_values,density = 1.):
        return super().__call__(scalar_cell_field,scalar_boundary_values,velocity_cell_field,velocity_boundary_values,density)
    
    
    @setup
    def initialise(self,scalar_cell_field,scalar_boundary_values,velocity_cell_field,velocity_boundary_values,density):
        self.mesh.to_warp()
        self.field_shape = scalar_cell_field.shape
        self.num_vars = self.field_shape[0]
        self.num_cells = self.field_shape[-1]
        self.internal_kernel,self.external_kernel = create_advection_kernel(self.num_vars,self.float_dtype)
        
        self.output_field = wp.empty_like(scalar_cell_field)


    
    def forward(self,scalar_cell_field,scalar_boundary_values,velocity_cell_field,velocity_boundary_values,density):
        
        wp.launch(kernel = self.internal_kernel,dim=self.num_cells,
                  inputs= [
                    scalar_cell_field,
                    velocity_cell_field,
                    density,
                    self.mesh.neighbors,
                    self.mesh.neighbors_offset,
                    self.mesh.face_normals,
                    self.mesh.cell_volumes
                  ],
                  outputs =[
                      self.output_field
                  ])
        
        wp.launch(kernel = self.external_kernel,
                  dim=len(self.mesh.exterior_faces),
                  inputs= [
                    scalar_boundary_values,
                    velocity_boundary_values,
                    density,
                    self.mesh.exterior_faces.cell_ids,
                    self.mesh.exterior_faces.normals,
                    self.mesh.cell_volumes
                  ],
                  outputs =[
                      self.output_field
                  ])


        return self.output_field
def create_advection_kernel(num_scalars,float_dtype):
    # Assume Density is constant for now
    scalar_vector = vector(num_scalars,float_dtype)
    vec3 = vector(3,float_dtype)
    
    @wp.func
    def upwind(owner_value:float_dtype,
               neighbor_value:float_dtype,
               mass_flow:float_dtype):
        # If mass flow is positive then
        return wp.where(mass_flow > 0.,owner_value,neighbor_value)
        
        
    @wp.kernel
    def advection_internal_face_kernel(
                        cell_field:wp.array2d[float_dtype],
                        velocity_field:wp.array2d[float_dtype],
                        density:float_dtype,  
                        cell_neighbors:wp.array1d[wp.vec2i],
                        cell_neighbors_offsets:wp.array1d[int],
                        face_normals:wp.array1d[vec3],
                        cell_volumes:wp.array1d[float_dtype],
                        output_values:wp.array2d[float_dtype]
                        ):
        
        
        cell_id = wp.tid()
        offset = cell_neighbors_offsets[cell_id]
        num_neighbors = cell_neighbors[offset][0]
        cell_volume = cell_volumes[cell_id]
        advec = scalar_vector()
        u_vec = vec3()
        for i in range(num_neighbors):
            neighbor_info = cell_neighbors[offset + 1 + i]
            neighbor_id = neighbor_info[0]
            face_id = neighbor_info[1]
            
            # Calculate velocity at face
            for jj in range(3):
                u_vec[jj] =  (velocity_field[jj,cell_id] + velocity_field[jj,neighbor_id])/2.
            
            face_normal = wp.where(cell_id < neighbor_id,1.,-1.)*face_normals[face_id] # If current cell is the neighbor cell of face then flip face normal
            # Face normal always points outwards from owner to neighbor
            mass_flow = density*wp.dot(face_normal,u_vec)
            
            for scalar in range(num_scalars):
                advec[scalar] = advec[scalar] + upwind(cell_field[scalar,cell_id],cell_field[scalar,neighbor_id],mass_flow)*mass_flow   
            
        advec /= cell_volume
        for scalar in range(num_scalars):
            output_values[scalar,cell_id] = advec[scalar]
            
        
        
        
    @wp.kernel
    def advection_external_face_kernel(
                                    boundary_value:wp.array2d[float_dtype],
                                    velocity_boundary:wp.array2d[float_dtype],
                                    density:float_dtype,
                                    boundary_face_to_cell_id:wp.array1d[int],
                                    face_normals:wp.array1d[vec3],
                                    cell_volumes:wp.array1d[float_dtype],
                                    output_values:wp.array2d[float_dtype]
    ):
        
        tid = wp.tid() # Loop through Boundary Faces!
        
        cell_id = boundary_face_to_cell_id[tid]
        u_vec = vec3()
        cell_volume = cell_volumes[cell_id]
        for jj in range(3):
            u_vec[jj] =  velocity_boundary[jj,tid]
        mass_flow = density*wp.dot(face_normals[tid],u_vec)/cell_volume # Cell_Id is always owner so normal always points away    
        
        for scalar in range(num_scalars):
            output_values[scalar,cell_id] = output_values[scalar,cell_id] + mass_flow*boundary_value[scalar,tid] # Hopefully avoid atomic
        
    
    return advection_internal_face_kernel,advection_external_face_kernel
import numpy as np
from pde_module.mesh import Mesh

class EulerianMesh(Mesh):
    def __init__(self, nodes, cells_connectivity, cell_types, dimension = None, float_dtype=np.float32, int_dtype=np.int32):
        super().__init__(nodes, cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
        self.faces.get_centroids_and_normals(self.nodes)
        self.faces.get_OwnerNeighbor()
        self.cells.get_centroids(self.nodes)
        self.cells.get_volumes(self.faces._cell_to_face_array_,self.faces._cell_to_face_offset_,self.faces.centroids,self.faces.normals)
        self.cells.get_neighbors(self.faces._cell_to_face_array_,self.faces._cell_to_face_offset_,self.faces.ownerNeighbor)

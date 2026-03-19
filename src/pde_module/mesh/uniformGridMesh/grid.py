import numpy as np
from pde_module.mesh.mesh import Mesh
from pde_module.mesh.cell_types import CELLTYPES_DICT,HEX,QUAD,EDGE
from math import prod
def get_ghost_cells(ghost_cells):
    ghost_cells = 0 if ghost_cells is None else ghost_cells
    assert isinstance(ghost_cells,int)
    assert ghost_cells >= 0  
    return ghost_cells

def add_ghost_coords(dx,coord_vectors,ghost_cells):
        if ghost_cells == 0:
            return coord_vectors
        # We need to add points dx
        coord_with_ghost = []
        for coord_vector in coord_vectors:
            if len(coord_vector) == 1: # If one point then leave alone
                coord_with_ghost.append(coord_vector)    
            else:
                left_g = np.array([coord_vector[0] - n*dx for n in range(1,ghost_cells+1)])
                right_g = np.array([coord_vector[-1] + n*dx for n in range(1,ghost_cells+1)])
                coord_with_ghost.append(np.concat((left_g,coord_vector,right_g),dtype=coord_vector.dtype))
        return coord_with_ghost



def create_nodes_grid(dx:float,num_points:tuple[int],origin:np.ndarray,ghost_cells:int):
    nodal_coordinates_vectors = tuple(np.arange(0,axis,dtype=origin.dtype)*dx - axis_origin for axis,axis_origin in zip(num_points,origin))
    ghost_coord_vectors = add_ghost_coords(dx,nodal_coordinates_vectors,ghost_cells)
    grid = np.meshgrid(*ghost_coord_vectors,indexing = 'ij')
    grid = np.stack(grid,axis = -1)
    return grid


def get_3d_connectivity(nx, ny, nz):
    # Create a 3D array of node IDs
    node_ids = np.arange(nx * ny * nz).reshape((nz, ny, nx))
    
    # Identify the 'base' nodes (bottom-front-left corner of each cell)
    # We exclude the last node in each dimension because they can't be base nodes
    base_nodes = node_ids[:-1, :-1, :-1].flatten()
    
    # Offsets to find the other 7 vertices relative to the base node
    # These depend on the strides of your node_ids array
    dy = nx
    dz = nx * ny
    
    # Ordering: 0, 1, 1+dy, dy (bottom) then +dz for the top face
    connectivity = np.vstack([
        base_nodes,
        base_nodes + 1,
        base_nodes + 1 + dy,
        base_nodes + dy,
        base_nodes + dz,
        base_nodes + 1 + dz,
        base_nodes + 1 + dy + dz,
        base_nodes + dy + dz
    ]).T # Cx8
    
    
    return connectivity

def get_2d_connectivity(nx, ny,*args):
    # 1. Create the grid of Node IDs
    # Assuming 'C' order: X increments fastest, then Y
    node_ids = np.arange(nx * ny).reshape((ny, nx))
    
    # 2. Identify 'base' nodes (bottom-left of each quad)
    # We exclude the last column and last row
    base_nodes = node_ids[:-1, :-1].flatten()
    
    # 3. Calculate offsets
    # The node to the right is +1
    # The node above is +nx
    dx = 1
    dy = nx
    
    # 4. Assemble connectivity (Counter-Clockwise)
    connectivity = np.vstack([
        base_nodes,             # 0: (i, j)
        base_nodes + dx,        # 1: (i+1, j)
        base_nodes + dx + dy,   # 2: (i+1, j+1)
        base_nodes + dy         # 3: (i, j+1)
    ]).T
    
    return connectivity


def get_1d_connectivity(nx,*args):
    # 1. Create a simple array of node IDs [0, 1, 2, ..., nx-1]
    node_ids = np.arange(nx)
    
    # 2. Pair each node 'i' with its neighbor 'i+1'
    # We take all nodes except the last as 'start' nodes
    # and all nodes except the first as 'end' nodes
    connectivity = np.vstack([
        node_ids[:-1], # Left nodes
        node_ids[1:]   # Right nodes
    ]).T
    
    return connectivity

def cell_connectivity_and_type(nodes_per_axis,num_cells,dimension,int_dtype = np.int32):
    connect_func = [get_1d_connectivity,get_2d_connectivity,get_3d_connectivity]
    cell_types = [EDGE,QUAD,HEX]

    connectivity,cell_type =(connect_func[dimension-1](*nodes_per_axis),cell_types[dimension-1] )
    num_nodes = np.full((len(connectivity),1),cell_type.num_nodes,dtype=int_dtype)
    connectivity = np.concatenate((num_nodes,connectivity),axis = -1,dtype= int_dtype)
    cell_type_arr = np.full(num_cells,cell_type.id,dtype= int_dtype)
    return connectivity.ravel(), cell_type_arr
    
class UniformGridMesh(Mesh):
    dx:float
    area:float
    volume:float
    ghost_cells:int
    nodal_grid: np.ndarray
    dimension: int
    origin : np.ndarray
    nodes_per_axis:tuple
    num_cells: np.ndarray
    
    
    def __init__(self,dx,nodes_per_axis:tuple[int],origin = None,ghost_cells = None,float_dtype = np.float32,int_dtype = np.int32):
        dx = float_dtype(dx)
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.ghost_cells = get_ghost_cells(ghost_cells)
        
        self.origin = np.zeros(3,dtype=float_dtype) if origin is None else np.array(origin,dtype=float_dtype)
        assert len(self.origin) == 3,'Origin must be a tuple of length 3'
        
        self.nodes_per_axis = nodes_per_axis
        self.num_cells = prod(i-1 for i in self.nodes_per_axis if i > 1)
        dimension = sum(1 for i in self.nodes_per_axis if i > 1)
        assert 1 <= dimension <= 3
        
        self.nodal_grid = create_nodes_grid(dx,self.nodes_per_axis,self.origin,self.ghost_cells)
        nodes = self.nodal_grid.reshape(-1,3)
        cells_connectivity,cell_types = cell_connectivity_and_type(self.nodes_per_axis,self.num_cells,dimension)
        super().__init__(nodes, cells_connectivity, cell_types, dimension, float_dtype, int_dtype)
    
    
    def create_structured_field(self,type:str,num_outputs:int):
        match type:
            case 'node':
                shape = self.nodal_grid.shape[0:-1] # We dont need the last axis
            case 'cell':
                shape = tuple(n-1 for n in self.nodal_grid.shape[:-1])
            case _:
                raise ValueError('Valid types are: node andcell')
        return np.array((*shape,num_outputs),dtype=self.float_dtype)
    
if __name__ == '__main__':
    mesh = UniformGridMesh(1,(10,10,10))
    pv_mesh = mesh.to_pyvista()
    pv_mesh.plot(show_edges=True)
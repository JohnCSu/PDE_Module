from numba import njit,int32
from numba import types
from numba.typed import List,Dict
import numpy as np
from .array_hash import numba_array_hash as array_hash

@njit()
def init_list_of_lists(n):
    outer = List()  # typed outer list
    for _ in range(n):
        inner = List.empty_list(types.int32)  # each inner list
        outer.append(inner)
    return outer

@njit()
def point_to_cells(cells,points):

    point_to_cells_list = init_list_of_lists(len(points))
    for cell_id, pts in enumerate(cells):
        for p in pts:
            point_to_cells_list[p].append(cell_id)

    return point_to_cells_list

@njit
def intersection(neighbor_candidates_from_points,cell_id):
    # Make a set of b for faster lookup -- NOt neccesarily as a and b are generally small so maybe not worth the memory copy???
    candidates_from_p0 = neighbor_candidates_from_points[0]
    
    neighbor_cell_id = -1

    for neighbor_cell_candidate in candidates_from_p0:
        if neighbor_cell_candidate == cell_id: # Skip candidate if candidate matches cell_id
            continue
        

        neighbor_cell_id = neighbor_cell_candidate

        for p in neighbor_candidates_from_points[1:]:
            if neighbor_cell_candidate not in p:
                neighbor_cell_id = -1
                break
        
        if neighbor_cell_id != -1:
            return neighbor_cell_id
        
    # If no Match Found
    return -1


@njit
def get_neighbors(cell_id:int,cells:np.ndarray[int,int],connectivity:list[list[int]],point_to_cells_list:list[list[int]]):
    '''
    Get all the neighbors of a cell
    '''
    neighbors = -1*np.ones(shape = len(connectivity),dtype= np.int32)
    
    for i,surface  in enumerate(connectivity):
        
        neighbor_candidates_from_points = List() # For each point, get all the connecting 

        for p in surface:
            neighbor_candidates_from_points.append( point_to_cells_list[cells[cell_id,p]])

        neighbor =  intersection(neighbor_candidates_from_points,cell_id)
        neighbors[i] =neighbor
        # print(neighbor)
    return neighbors


@njit
def get_all_neighbors(cells,points,point_to_cells_list,connectivity):
    num_faces = len(connectivity)
    num_cells = len(cells)
    num_points = points
    neighbors = np.empty(shape = (num_cells,num_faces),dtype = np.int32 )
    for i in range(num_cells):
        neighbors[i] = get_neighbors(i,cells,connectivity,point_to_cells_list)

    return neighbors


@njit
def compute_interior_face_adjacency(neighbors:np.ndarray):
    N,M = neighbors.shape
    cell_ids = np.repeat(np.arange(N,dtype = np.int32), M)
    # ,cell_face_idx = np.indices((N,M),dtype = neighbors.dtype)
    neighbor_ids = neighbors.ravel()

    cell_adjacency = np.column_stack((cell_ids,neighbor_ids))

    interior_face_mask = (cell_adjacency[:,1] != -1)

    interior_adjacency = cell_adjacency[interior_face_mask]

    interior_adjacency = sort_array(interior_adjacency)
    # interior_adjacency = np.unique(interior_adjacency,axis=0)
    interior_adjacency = unique_rows_of_array(interior_adjacency)


    if np.any(interior_adjacency < 0):
        raise ValueError('Interior Faces must have an Owner and Neighbor')

    return interior_adjacency



# @njit Doesnt need JIt compilation
def compute_cell_face_index(neighbors,adjaceny,is_interior = True):
    N,M = neighbors.shape
    _,cell_face_idx = np.indices((N,M),dtype = neighbors.dtype)
    if is_interior:
        return cell_face_idx[adjaceny]
    else:
        exterior_cell_face_idx = cell_face_idx[adjaceny[:,0]]
        return np.vstack(exterior_cell_face_idx,-1*np.ones_like(exterior_cell_face_idx))
    

@njit
def compute_exterior_face_adjaceny(neighbors):
    '''Find out the cell that own exterior Face'''
    N,M = neighbors.shape
    cell_ids = np.repeat(np.arange(N,dtype = np.int32), M)

    neighbor_ids = neighbors.ravel()

    # cell_adjacency = np.column_stack((cell_ids,neighbor_ids))

    exterior_face_mask = (neighbor_ids[:,1] == -1)
    
    exterior_adjacency = np.column_stack((cell_ids[exterior_face_mask],-1*np.ones_like(cell_ids)))

    return exterior_adjacency
    



@njit
def unique_rows_of_array(arr):
    # We allocate memory of same size (could be all disjointed meshs)
    assert len(arr.shape) == 2, 'unique_rows_of_array only works for NxM arrays'

    n_cols = arr.shape[-1]
    
    d = Dict()

    for row in arr:
        key = array_hash(row)
        if key not in d:
            d[key] = row
    # Create unique array
    unique_arr = np.empty(shape = (len(d),n_cols),dtype= np.int32)
    for i,key in enumerate(d):
        unique_arr[i] = d[key]
    return unique_arr


@njit
def sort_array(arr:np.ndarray):
    '''Sort a NxM'''
    if len(arr.shape) != 2:
        raise ValueError('Array ndim must be equal to 2')
    for i in range(len(arr)):
        arr[i] = np.sort(arr[i])
    return arr


@njit
def combine_arrays(a: np.ndarray, b:np.ndarray):
    return np.concatenate((a, b))



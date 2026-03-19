import numpy as np 
import numba as nb
from pde_module.mesh.cell_types.cell_types import LOCAL_EDGE_ORDERING_DICT,CELLTYPES_DICT
from pde_module.mesh.cell import Cells

def get_edges(cells:Cells):    
    max_num_edges = max([CELLTYPES_DICT[key].num_edges for key in cells.unique_cell_types])
    raw_edges = get_raw_edges(cells.connectivity, cells.IDs,cells.types,max_num_edges,LOCAL_EDGE_ORDERING_DICT)
    edges = np.unique(raw_edges,axis = 0) # This sorts the array so the -1,-1 edge is guranteed to be first
    if sum(edges[0]) == -2:# Ignore the first one
        edges = edges[1:]
    return edges

@nb.njit
def get_raw_edges(cell_connectivity, cell_offsets,cell_types,num_max_edges,local_edge_dict:dict[str,np.ndarray[int,int]]):
    '''
    Return a array of Raw ordered pairs of edges where e[0] <= e[1]
    '''
    raw_edges = np.full((len(cell_offsets),num_max_edges,2),-1)
    for id in nb.prange(len(cell_offsets)):
        offset = cell_offsets[id]
        num_nodes = cell_connectivity[offset]
        nodes = cell_connectivity[offset+1: offset+1+num_nodes]
        cell_type = cell_types[id]
        local_edge_ordering = local_edge_dict[cell_type] # N,2 array
        for j in range(local_edge_ordering.shape[0]):
            edge = nodes[local_edge_ordering[j]]
            raw_edges[id,j,0],raw_edges[id,j,1] = np.min(edge),np.max(edge)
    
    return raw_edges.reshape(-1,2)
    
    
    
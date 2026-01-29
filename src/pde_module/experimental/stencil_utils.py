import warp as wp
from warp.types import vector,matrix

def create_stencil_op(input_vector:vector,stencil:vector,ghost_cells:int):
    '''
    Simple Stencil Op that gathers the neighbors of the current index based on the given axis
    and calculates the stencil. Outputs a vector of same type and length as input_vector
    
    TODO:
    - We can vectorise the operation by gathering the vectors into a matrix (by cols) and then matmul with stencil
    '''
    assert wp.types.type_is_vector(input_vector)
    length = stencil._length_
    assert (length % 2) == 1,'stencil must be odd sized'
    max_shift = (length -1)//2
    
    
    assert max_shift <= ghost_cells ,'Max shift must be <=  ghost cell to avoid out of bounds array access!'
    @wp.func
    def stencil_op(input_values:wp.array3d(dtype=input_vector),
                   index:wp.vec3i,
                   stencil:type(stencil),
                   axis:int):
        
        value = input_vector()
        for i in range(wp.static(length)):
            shift = i - max_shift
            stencil_index = index
            stencil_index[axis] = index[axis] + shift
            value += input_values[stencil_index[0],stencil_index[1],stencil_index[2]]*stencil[i]

        return value
            
    return stencil_op


def eligible_dims_and_shift(grid_shape,ghost_cells):
    '''
    Return dims that have more than 1 points in that direction as a vector and also the corresponding shift in each
    eligible dim due to ghost_cells
    
    E.g (3,5,1) will return (0,1) indicating axes 0 and 1 are the dimensions of our grid
    if ghost cells is non_zero then we the second vector is [ghost_cells,ghost_cells,0]
    '''
    d = tuple(i for i,x in enumerate(grid_shape) if x > 1 ) # Store dimensions that are eligible
    return wp.types.vector(length=len(d),dtype = int)(d) , wp.vec3i([ghost_cells if x > 1 else 0 for x in grid_shape])


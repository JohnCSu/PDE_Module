import warp as wp

def boundary(kernel,
            current_values:wp.array4d(),
            threads_shape:tuple[int],
            boundary_indices:wp.array2d(dtype=int),
            boundary_type:wp.array(dtype=int),
            boundary_value:wp.array(dtype=float),
            interior_indices: wp.array(dtype = int),
            interior_adjaceny:wp.array(dtype = wp.vec3i),
            levels:wp.array(dtype=int),
            new_values:wp.array4d()):    
    '''
    Functional Version of boundary module. Only Applies boundary values and not ghost cell corections
    '''
    wp.launch(kernel,dim = threads_shape, inputs = [current_values,boundary_indices,boundary_type,boundary_value,interior_indices,interior_adjaceny,levels],outputs=[new_values])
    return new_values
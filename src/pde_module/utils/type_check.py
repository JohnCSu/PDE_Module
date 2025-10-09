import warp as wp

def is_dtype_wp_vector(arr:wp.array):
    '''
    Returns a bool on whether the passed in warp array's dtype is a vector (True if dtype is a vec, False otherwise)
    '''    
    assert isinstance(arr,wp.array)
    return is_wp_vector(arr.dtype)


def is_wp_vector(vec):
    '''
    Check if passed in object is a wp.vector and return True. False otherwise
    '''
    if hasattr(vec,'_wp_generic_type_str_'):
        if vec._wp_generic_type_str_ == 'vec_t':
            return True

    return False
    
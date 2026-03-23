import warp as wp
from warp.types import vector,matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix

def wp_array_to_vec_or_mat(wp_arr:wp_Array) -> wp_Vector|wp_Matrix:
    arr = wp_arr.numpy()
    if len(arr.shape) == 1:
        return vector(len(arr),dtype = wp.dtype_from_numpy(arr.dtype))(arr)
    elif len(arr.shape) == 2:
        return matrix(arr.shape,dtype = wp.dtype_from_numpy(arr.dtype))(arr)
    else:
        raise ValueError('only arrays that map to 1D or 2D numpy arrays can be used')


